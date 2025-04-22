#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

import common_utils.features as features
import common_utils.kaldi_data as kaldi_data
import numpy as np
import torch
from typing import Tuple
import logging
import soundfile as sf
import os

def _count_frames(data_len: int, size: int, step: int) -> int:
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)


def _gen_frame_indices(
    data_length: int,
    size: int,
    step: int,
    use_last_samples: bool,
    min_length: int,
) -> None:
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step > min_length:
            yield (i + 1) * step, data_length


class KaldiDiarizationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        chunk_size: int,
        context_size: int,
        feature_dim: int,
        frame_shift: int,
        frame_size: int,
        input_transform: str,
        n_speakers: int,
        sampling_rate: int,
        shuffle: bool,
        subsampling: int,
        use_last_samples: bool,
        min_length: int,
        dtype: type = np.float32,
        cache_dir: str = '.cache',
        use_cache: bool = True,
        create_cache: bool = True,
    ):
        self.data_dir = data_dir
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.feature_dim = feature_dim
        self.subsampling = subsampling
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.sampling_rate = sampling_rate
        self.chunk_indices = []

        self.data = kaldi_data.KaldiData(self.data_dir)

        # 캐시 관련 변수 선언
        self.cache_dir = os.path.join(data_dir, cache_dir)
        self.use_cache = use_cache
        self.create_cache = create_cache

        # 캐시 디렉토리 생성
        if self.use_cache or self.create_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # make chunk indices: filepath, start_frame, end_frame
        for rec in self.data.wavs:
            with sf.SoundFile(self.data.wavs[rec]) as audio_file:
                duration = len(audio_file) / audio_file.samplerate
            data_len = int(
                duration * sampling_rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            if chunk_size > 0:
                for st, ed in _gen_frame_indices(
                        data_len,
                        chunk_size,
                        chunk_size,
                        use_last_samples,
                        min_length
                ):
                    self.chunk_indices.append(
                        (rec, st * self.subsampling, ed * self.subsampling))
            else:
                self.chunk_indices.append(
                    (rec, 0, data_len * self.subsampling))
        logging.info(f"#files: {len(self.data.wavs)}, "
                     "#chunks: {len(self.chunk_indices)}")

        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.chunk_indices)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        rec, st, ed = self.chunk_indices[i]

        # 캐시 파일 경로 생성
        cache_key = f"{rec}_{st}_{ed}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npz")

        # use_cache가 True이고 캐시 파일이 존재하는 경우 캐시에서 데이터 로드
        if self.use_cache and os.path.exists(cache_file):
            data = np.load(cache_file)
            return torch.from_numpy(data['Y']), torch.from_numpy(data['T']), str(data['rec'][0])

        # 캐시가 없거나 use_cache가 False인 경우 원본 데이터 처리
        Y, T = features.get_labeledSTFT(
            self.data,
            rec,
            st,
            ed,
            self.frame_size,
            self.frame_shift,
            self.n_speakers
        )
        Y = features.transform(
            Y, self.sampling_rate, self.feature_dim, self.input_transform)
        Y_spliced = features.splice(Y, self.context_size)
        Y_ss, T_ss = features.subsample(Y_spliced, T, self.subsampling)

        # If the sample contains more than "self.n_speakers" speakers,
        #  extract top-(self.n_speakers) speakers
        if self.n_speakers and T_ss.shape[1] > self.n_speakers:
            selected_spkrs = np.argsort(
                T_ss.sum(axis=0))[::-1][:self.n_speakers]
            T_ss = T_ss[:, selected_spkrs]
            
        # create_cache가 True인 경우 캐시 파일 생성
        if self.create_cache:
            np.savez(
                cache_file,
                Y=Y_ss,
                T=T_ss,
                rec=np.array([rec])
            )

        return torch.from_numpy(np.copy(Y_ss)), torch.from_numpy(
            np.copy(T_ss)), rec
