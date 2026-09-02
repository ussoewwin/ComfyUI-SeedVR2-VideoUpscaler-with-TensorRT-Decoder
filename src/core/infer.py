# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import List, Optional, Tuple, Union
import os
import torch
from einops import rearrange
from omegaconf import DictConfig, ListConfig
from torch import Tensor
from ..common.diffusion import (
    classifier_free_guidance_dispatcher,
    create_sampler_from_config,
    create_sampling_timesteps_from_config,
    create_schedule_from_config,
)
from ..common.distributed import (
    get_device,
)
from ..optimization.performance import (
    optimized_channels_to_last,
    optimized_channels_to_second
)
from ..models.dit_3b import na


# Remembers the pre-pad spatial size so the decoder can crop back to the original
# dimensions. Spatial dims that are not multiples of 8 make tile boundaries drift
# from latent boundaries, producing smeared (top-left) / black (bottom-right) tiles.
_TRT_CROP_HW = [-1, -1]


def _debug_save_latent(latent, tag: str) -> None:
    """Debug hook: save post-scale latent (SEEDVR2_SAVE_LATENT_DIR) and log stats."""
    import os as _os
    d = _os.environ.get("SEEDVR2_SAVE_LATENT_DIR")
    if d:
        from pathlib import Path as _P
        _P(d).mkdir(parents=True, exist_ok=True)
        torch.save(latent.float().cpu(), _P(d) / f"latent_{tag}.pt")
    if _os.environ.get("SEEDVR2_LATENT_STATS"):
        f = latent.float()
        print(f"[SeedVR2][latent:{tag}] shape={tuple(latent.shape)} "
              f"min={f.min().item():.4f} max={f.max().item():.4f} "
              f"mean={f.mean().item():.4f} std={f.std().item():.4f} "
              f"nan={torch.isnan(f).any().item()} inf={torch.isinf(f).any().item()}", flush=True)


def _trt_encode_batch(enc_sample, vae, dit_model, engine_frames_setting):
    """Encode a long clip by feeding the TensorRT encoder 29f-sized chunks ONLY.

    The DiT batch (e.g. 185 frames) is split here into engine-sized chunks with a
    4-frame temporal overlap; the encoder never receives more than engine_frames.
    """
    from .trt_encoder import encode as trt_encode, resolve_engine_frames
    import os as _os
    _sd = _os.environ.get("SEEDVR2_SAVE_LATENT_DIR")
    if _sd:
        from pathlib import Path as _P
        _P(_sd).mkdir(parents=True, exist_ok=True)
        torch.save(enc_sample.float().cpu(), _P(_sd) / "enc_input_raw.pt")
    total = enc_sample.shape[2]
    # Pad spatial dims to multiples of 8 so tile boundaries align with latent boundaries.
    h, w = enc_sample.shape[3], enc_sample.shape[4]
    ph = ((h + 7) // 8) * 8
    pw = ((w + 7) // 8) * 8
    if (ph, pw) != (h, w):
        enc_sample = torch.nn.functional.pad(enc_sample, (0, pw - w, 0, ph - h))
        _TRT_CROP_HW[0], _TRT_CROP_HW[1] = h, w
    else:
        _TRT_CROP_HW[0], _TRT_CROP_HW[1] = -1, -1
    if _sd:
        torch.save(enc_sample.float().cpu(), _P(_sd) / "enc_input_padded.pt")
    engine_frames = resolve_engine_frames(engine_frames_setting)
    if engine_frames is None:
        raise RuntimeError("No TensorRT VAE encoder engine available")
    if total == engine_frames:
        return trt_encode(enc_sample, vae=vae, dit_model=dit_model, engine_frames=str(engine_frames))
    # Stride must stay a multiple of 4 so chunk boundaries align with latent-frame
    # boundaries. A non-multiple stride (e.g. 81) produces a latent whose 4-frame
    # window is missing its leading frames -> corrupt/black frames after decode.
    stride = ((engine_frames - 4) // 4) * 4
    if stride < 4:
        stride = 4
    lat_parts = []
    starts = list(range(0, total - engine_frames + 1, stride))
    if starts[-1] != total - engine_frames:
        starts.append(total - engine_frames)
    for start in starts:
        chunk = enc_sample[:, :, start:start + engine_frames].contiguous()
        lat = trt_encode(chunk, vae=vae, dit_model=dit_model, engine_frames=str(engine_frames))
        lat_parts.append((lat, start // 4))
    lat_total = (total - 1) // 4 + 1
    lat0 = lat_parts[0][0]
    latent = torch.zeros((1, 16, lat_total, lat0.shape[3], lat0.shape[4]), device=lat0.device, dtype=lat0.dtype)
    # Causal encoder: a chunk's leading latents (context-poor) are LESS accurate than
    # the previous chunk's trailing latents (full context). So earlier chunks win.
    # Write in reverse so the first chunk keeps its (accurate) values.
    for lat, lat_start in reversed(lat_parts):
        latent[:, :, lat_start:lat_start + lat.shape[2]] = lat
    return latent


def _trt_decode_batch(dec_latent, vae, dit_model, engine_frames_setting):
    """Decode a long latent by feeding the TensorRT decoder engine-sized chunks ONLY."""
    from .trt_decoder import decode as trt_decode, resolve_engine_frames
    latent_frames = dec_latent.shape[2]
    engine_video_frames = resolve_engine_frames(engine_frames_setting)
    if engine_video_frames is None:
        raise RuntimeError("No TensorRT VAE decoder engine available")
    engine_latent = (engine_video_frames - 1) // 4 + 1
    if latent_frames == engine_latent:
        return trt_decode(dec_latent, vae=vae, dit_model=dit_model, engine_frames=str(engine_video_frames))
    lat_stride = engine_latent - 1
    parts = []
    starts = list(range(0, latent_frames - engine_latent + 1, lat_stride))
    if starts[-1] != latent_frames - engine_latent:
        starts.append(latent_frames - engine_latent)
    for start in starts:
        chunk = dec_latent[:, :, start:start + engine_latent].contiguous()
        sample = trt_decode(chunk, vae=vae, dit_model=dit_model, engine_frames=str(engine_video_frames))
        parts.append((sample, start * 4))
    out_frames = (latent_frames - 1) * 4 + 1
    s0 = parts[0][0]
    result = torch.zeros((1, 3, out_frames, s0.shape[3], s0.shape[4]), device=s0.device, dtype=s0.dtype)
    for sample, out_start in parts:
        result[:, :, out_start:out_start + engine_video_frames] = sample
    # No crop back: the upscaler rounds the target output dims to multiples of 8
    # (see generation_utils.prepare_video_transforms), so the padded size IS the
    # target size. Cropping would only lose resolution.
    return result


class VideoDiffusionInfer():
    def __init__(self, config: DictConfig, debug: 'Debug',
                 encode_tiled: bool = False, encode_tile_size: Tuple[int, int] = (512, 512), 
                 encode_tile_overlap: Tuple[int, int] = (64, 64),
                 decode_tiled: bool = False, decode_tile_size: Tuple[int, int] = (512, 512),
                 decode_tile_overlap: Tuple[int, int] = (64, 64),
                 tile_debug: str = "false",
                 use_tensorrt_vae: bool = False):
        self.config = config
        self.debug = debug
        # Store separate encode and decode tiling parameters
        self.encode_tiled = encode_tiled
        self.encode_tile_size = encode_tile_size
        self.encode_tile_overlap = encode_tile_overlap
        self.decode_tiled = decode_tiled
        self.decode_tile_size = decode_tile_size
        self.decode_tile_overlap = decode_tile_overlap
        self.tile_debug = tile_debug
        self.use_tensorrt_vae = use_tensorrt_vae
        
    def _resolve_dit_name(self) -> Optional[str]:
        """Return the DiT checkpoint filename selected by the workflow DiT loader."""
        cp = getattr(self, "_dit_checkpoint", None)
        if cp:
            name = str(cp).replace("\\", "/").rsplit("/", 1)[-1]
            if name:
                return name
        return None



    def get_condition(self, latent: Tensor, latent_blur: Tensor, task: str) -> Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if task == "t2v" or t == 1:
            # t2i or t2v generation.
            if task == "sr":
                cond[:, ..., :-1] = latent_blur[:]
                cond[:, ..., -1:] = 1.0
            return cond
        if task == "i2v":
            # i2v generation.
            cond[:1, ..., :-1] = latent[:1]
            cond[:1, ..., -1:] = 1.0
            return cond
        if task == "v2v":
            # v2v frame extension.
            cond[:2, ..., :-1] = latent[:2]
            cond[:2, ..., -1:] = 1.0
            return cond
        if task == "sr":
            # sr generation.
            cond[:, ..., :-1] = latent_blur[:]
            cond[:, ..., -1:] = 1.0
            return cond
        raise NotImplementedError
    
    def configure_diffusion(self, device: Optional[torch.device] = None, dtype=torch.float32):
        """
        Configure diffusion schedule and sampler.
        
        Args:
            device: Device for schedule tensors. If None, uses get_device()
            dtype: Data type for computations
        """
        # Use provided device or fallback to standard detection
        if device is None:
            device = get_device()
        elif not isinstance(device, torch.device):
            device = torch.device(device)
            
        self.schedule = create_schedule_from_config(
            config=self.config.diffusion.schedule,
            device=device,
            dtype=dtype,
        )
        self.sampling_timesteps = create_sampling_timesteps_from_config(
            config=self.config.diffusion.timesteps.sampling,
            schedule=self.schedule,
            device=device,
            dtype=dtype,
        )
        self.sampler = create_sampler_from_config(
            config=self.config.diffusion.sampler,
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
        )
        # Propagate debug to sampler
        if hasattr(self, 'debug'):
            self.sampler.debug = self.debug

    # -------------------------------- Helper ------------------------------- #

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor]) -> List[Tensor]:
        """VAE encode with configured dtype - converts samples to latents with optional tiling"""
        use_sample = self.config.vae.get("use_sample", True)
        latents = []
        if len(samples) > 0:
            # Use VAE model's current device
            # This ensures consistency with where the VAE model is loaded
            try:
                device = next(self.vae.parameters()).device
            except StopIteration:
                # Fallback if VAE has no parameters (shouldn't happen)
                device = get_device()
            
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group samples of the same shape to batches if enabled.
            if self.config.vae.grouping:
                batches, indices = na.pack(samples)
            else:
                batches = [sample.unsqueeze(0) for sample in samples]

            # VAE process by each group.
            for sample in batches:
                # Check TensorRT VAE encoder
                if getattr(self, "use_tensorrt_vae", False) or os.environ.get("SEEDVR2_TRT_ENCODER", "0") == "1":
                    try:
                        from .trt_encoder import is_available as trt_enc_available, encode as trt_encode
                        enc_sample = sample if sample.ndim == 5 else sample.unsqueeze(0)
                        if enc_sample.ndim == 5 and trt_enc_available(enc_sample.shape[2]):
                            self.debug.log(f"Encoding with TensorRT VAE Encoder (engine={getattr(self, 'use_tensorrt_engine_frames', 'auto')})", category="info", indent_level=1)
                            latent = _trt_encode_batch(enc_sample, self.vae, self._resolve_dit_name(), getattr(self, 'use_tensorrt_engine_frames', 'auto'))
                            latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
                            latent = optimized_channels_to_last(latent)
                            latent = (latent - shift) * scale
                            _debug_save_latent(latent, "trt")
                            latents.append(latent)
                            continue
                    except Exception as trt_err:
                        self.debug.log(f"TensorRT VAE Encoder fallback to standard VAE: {trt_err}", category="warning", indent_level=1)

                if hasattr(self.vae, "preprocess"):
                    sample = self.vae.preprocess(sample)

                # Detect VAE model dtype
                try:
                    vae_dtype = next(self.vae.parameters()).dtype
                except StopIteration:
                    vae_dtype = dtype  # Fallback

                # Use autocast if VAE dtype differs from input dtype
                # Skip autocast on MPS (only supports bf16, unified memory = no benefit)
                # Instead, explicitly convert input to model dtype
                import gc
                latent_chunks = []
                num_frames = sample.size(0)

                for f_idx in range(num_frames):
                    frame_sample = sample[f_idx:f_idx+1]
                    
                    if vae_dtype != frame_sample.dtype:
                        if device.type == 'mps':
                            # MPS: explicit dtype conversion instead of autocast
                            frame_sample = frame_sample.to(vae_dtype)
                            if use_sample:
                                frame_latent = self.vae.encode(frame_sample, tiled=self.encode_tiled, tile_size=self.encode_tile_size, 
                                                        tile_overlap=self.encode_tile_overlap).latent
                            else:
                                frame_latent = self.vae.encode(frame_sample, tiled=self.encode_tiled, tile_size=self.encode_tile_size,
                                                    tile_overlap=self.encode_tile_overlap).posterior.mode().squeeze(2)
                        else:
                            with torch.autocast(device.type, frame_sample.dtype, enabled=True):
                                if use_sample:
                                    frame_latent = self.vae.encode(frame_sample, tiled=self.encode_tiled, tile_size=self.encode_tile_size, 
                                                            tile_overlap=self.encode_tile_overlap).latent
                                else:
                                    frame_latent = self.vae.encode(frame_sample, tiled=self.encode_tiled, tile_size=self.encode_tile_size,
                                                        tile_overlap=self.encode_tile_overlap).posterior.mode().squeeze(2)
                    else:
                        if use_sample:
                            frame_latent = self.vae.encode(frame_sample, tiled=self.encode_tiled, tile_size=self.encode_tile_size, 
                                                    tile_overlap=self.encode_tile_overlap).latent
                        else:
                            # Deterministic vae encode, only used for i2v inference (optionally)
                            frame_latent = self.vae.encode(frame_sample, tiled=self.encode_tiled, tile_size=self.encode_tile_size,
                                                tile_overlap=self.encode_tile_overlap).posterior.mode().squeeze(2)

                    latent_chunks.append(frame_latent)
                    
                    del frame_sample
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()

                latent = torch.cat(latent_chunks, dim=0)
                del latent_chunks

                latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
                latent = optimized_channels_to_last(latent)
                latent = (latent - shift) * scale
                _debug_save_latent(latent, "fp16")
                latents.append(latent)

            # Ungroup back to individual latent with the original order.
            if self.config.vae.grouping:
                latents = na.unpack(latents, indices)
            else:
                latents = [latent.squeeze(0) for latent in latents]
            
            self.debug.log(f"Latents shape: {latents[0].shape}", category="info", indent_level=1)

        return latents
    

    @torch.no_grad()
    def vae_decode(self, latents: List[Tensor]) -> List[Tensor]:
        """VAE decode with configured dtype - converts latents to samples with optional tiling"""
        samples = []
        if len(latents) > 0:
            # Use VAE model's current device
            # This ensures consistency with where the VAE model is loaded
            try:
                device = next(self.vae.parameters()).device
            except StopIteration:
                # Fallback if VAE has no parameters (shouldn't happen)
                device = get_device()
            
            dtype = getattr(torch, self.config.vae.dtype)
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group samples of the same shape to batches if enabled.
            if self.config.vae.grouping:
                latents, indices = na.pack(latents)
            else:
                latents = [latent.unsqueeze(0) for latent in latents]

            self.debug.log(f"Latents shape: {latents[0].shape}", category="info", indent_level=1)

            for i, latent in enumerate(latents):
                latent = latent / scale + shift
                latent = optimized_channels_to_second(latent)
                latent = latent.squeeze(2)

                # Check TensorRT VAE decoder
                if getattr(self, "use_tensorrt_vae", False) or os.environ.get("SEEDVR2_TRT_DECODER", "0") == "1":
                    try:
                        from .trt_decoder import is_available as trt_dec_available, decode as trt_decode
                        dec_latent = latent if latent.ndim == 5 else latent.unsqueeze(0)
                        if dec_latent.ndim == 5 and trt_dec_available(dec_latent.shape[2]):
                            self.debug.log(f"Decoding with TensorRT VAE Decoder (engine={getattr(self, 'use_tensorrt_decode_engine_frames', 'auto')})", category="info", indent_level=1)
                            sample = _trt_decode_batch(dec_latent, self.vae, self._resolve_dit_name(), getattr(self, 'use_tensorrt_decode_engine_frames', 'auto'))
                            if sample.ndim == 5 and sample.shape[0] == 1:
                                sample = sample.squeeze(0)
                            samples.append(sample)
                            continue
                    except Exception as trt_dec_err:
                        self.debug.log(f"TensorRT VAE Decoder fallback to standard VAE: {trt_dec_err}", category="warning", indent_level=1)

                # Detect VAE model dtype
                try:
                    vae_dtype = next(self.vae.parameters()).dtype
                except StopIteration:
                    vae_dtype = dtype  # Fallback

                # Use autocast if VAE dtype differs from latent dtype
                # Skip autocast on MPS (only supports bf16, unified memory = no benefit)
                import gc
                sample_chunks = []
                num_frames = latent.size(0)
                self.debug.log(f"Decoding {num_frames} frames iteratively to prevent OOM...", category="vae", indent_level=2)

                for f_idx in range(num_frames):
                    frame_latent = latent[f_idx:f_idx+1]
                    
                    if vae_dtype != frame_latent.dtype:
                        if device.type == 'mps':
                            # MPS: explicit dtype conversion instead of autocast
                            frame_latent = frame_latent.to(vae_dtype)
                            frame_sample = self.vae.decode(
                                frame_latent,
                                tiled=self.decode_tiled, tile_size=self.decode_tile_size,
                                tile_overlap=self.decode_tile_overlap
                            ).sample
                        else:
                            with torch.autocast(device.type, frame_latent.dtype, enabled=True):
                                frame_sample = self.vae.decode(
                                    frame_latent,
                                    tiled=self.decode_tiled, tile_size=self.decode_tile_size,
                                    tile_overlap=self.decode_tile_overlap
                                ).sample
                    else:
                        frame_sample = self.vae.decode(
                            frame_latent,
                            tiled=self.decode_tiled, tile_size=self.decode_tile_size,
                            tile_overlap=self.decode_tile_overlap
                        ).sample

                    sample_chunks.append(frame_sample)
                    
                    del frame_latent
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()

                sample = torch.cat(sample_chunks, dim=0)
                del sample_chunks

                if hasattr(self.vae, "postprocess"):
                    sample = self.vae.postprocess(sample)

                samples.append(sample)

            if self.config.vae.grouping:
                samples = na.unpack(samples, indices)
            else:
                samples = [sample.squeeze(0) for sample in samples]

        return samples


    def timestep_transform(self, timesteps: Tensor, latents_shapes: Tensor):
        # Skip if not needed.
        if not self.config.diffusion.timesteps.get("transform", False):
            return timesteps

        # Compute resolution.
        vt = self.config.vae.model.get("temporal_downsample_factor", 4)
        vs = self.config.vae.model.get("spatial_downsample_factor", 8)
        frames = (latents_shapes[:, 0] - 1) * vt + 1
        heights = latents_shapes[:, 1] * vs
        widths = latents_shapes[:, 2] * vs

        # Compute shift factor.
        def get_lin_function(x1, y1, x2, y2):
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return lambda x: m * x + b

        img_shift_fn = get_lin_function(x1=256 * 256, y1=1.0, x2=1024 * 1024, y2=3.2)
        vid_shift_fn = get_lin_function(x1=256 * 256 * 37, y1=1.0, x2=1280 * 720 * 145, y2=5.0)
        shift = torch.where(
            frames > 1,
            vid_shift_fn(heights * widths * frames),
            img_shift_fn(heights * widths),
        )

        # Shift timesteps.
        timesteps = timesteps / self.schedule.T
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * self.schedule.T
        return timesteps


    @torch.no_grad()
    def inference(
        self,
        noises: List[Tensor],
        conditions: List[Tensor],
        texts_pos: Union[List[str], List[Tensor], List[Tuple[Tensor]]],
        texts_neg: Union[List[str], List[Tensor], List[Tuple[Tensor]]],
        cfg_scale: Optional[float] = None,
    ) -> List[Tensor]:
        assert len(noises) == len(conditions) == len(texts_pos) == len(texts_neg)
        batch_size = len(noises)

        # Return if empty.
        if batch_size == 0:
            return []
        
        # Set cfg scale
        if cfg_scale is None:
            cfg_scale = self.config.diffusion.cfg.scale
        
        # Text embeddings.
        assert type(texts_pos[0]) is type(texts_neg[0])
        if isinstance(texts_pos[0], str):
            text_pos_embeds, text_pos_shapes = self.text_encode(texts_pos)
            text_neg_embeds, text_neg_shapes = self.text_encode(texts_neg)
        elif isinstance(texts_pos[0], tuple):
            text_pos_embeds, text_pos_shapes = [], []
            text_neg_embeds, text_neg_shapes = [], []
            for pos in zip(*texts_pos):
                emb, shape = na.flatten(pos)
                text_pos_embeds.append(emb)
                text_pos_shapes.append(shape)
            for neg in zip(*texts_neg):
                emb, shape = na.flatten(neg)
                text_neg_embeds.append(emb)
                text_neg_shapes.append(shape)
        else:
            text_pos_embeds, text_pos_shapes = na.flatten(texts_pos)
            text_neg_embeds, text_neg_shapes = na.flatten(texts_neg)
        
        # Flatten.
        latents, latents_shapes = na.flatten(noises)
        latents_cond, _ = na.flatten(conditions)
        
        latents = self.sampler.sample(
            x=latents,
            f=lambda args: classifier_free_guidance_dispatcher(
                pos=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_pos_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_pos_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                neg=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_neg_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_neg_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                scale=(
                    cfg_scale
                    if (args.i + 1) / len(self.sampler.timesteps)
                    <= self.config.diffusion.cfg.get("partial", 1)
                    else 1.0
                ),
                rescale=self.config.diffusion.cfg.rescale,
            ),
        )

        latents = na.unflatten(latents, latents_shapes)

        # Clean up temporary tensors
        del latents_cond
        del latents_shapes
        del text_pos_embeds
        del text_neg_embeds
        del text_pos_shapes
        del text_neg_shapes
            
        return latents