@torch.no_grad()
def compute_talking_scores(
        self,
        audio_data,
        sample_rate,
        faces,
    ):

        audio = self._prepare_audio(
            audio_data,
            sample_rate
        )

        video = self._prepare_faces(
            faces
        )

        if audio is None:
            return -10.0, None

        if video is None:
            return -10.0, None

        try:

            audio_tensor = torch.from_numpy(
                audio
            ).float().to(self.device)

            video_tensor = torch.from_numpy(
                video
            ).float().to(self.device)

            # TalkNet:
            # 100 audio frames -> 25 video frames

            video_frames = video_tensor.shape[0]

            num_video_frames = (
                video_frames // 25
            ) * 25

            if num_video_frames < 25:
                return -10.0, None

            video_tensor = video_tensor[
                :num_video_frames
            ]

            required_audio_frames = (
                num_video_frames * 4
            )

            if audio_tensor.shape[0] < required_audio_frames:
                return -10.0, None

            audio_tensor = audio_tensor[
                :required_audio_frames
            ]

            scores = []
            logits_list = []

            num_chunks = (
                num_video_frames // 25
            )

            for chunk_idx in range(num_chunks):

                audio_start = (
                    chunk_idx * 100
                )

                audio_end = (
                    audio_start + 100
                )

                audio_chunk = (
                    audio_tensor[
                        audio_start:audio_end
                    ]
                    .unsqueeze(0)
                )

                video_start = (
                    chunk_idx * 25
                )

                video_end = (
                    video_start + 25
                )

                video_chunk = (
                    video_tensor[
                        video_start:video_end
                    ]
                    .unsqueeze(0)
                )

                # -------------------------------
                # Audio frontend
                # -------------------------------

                audio_embed = (
                    self.model
                    .forward_audio_frontend(
                        audio_chunk
                    )
                )

                # -------------------------------
                # Visual frontend
                # -------------------------------

                visual_embed = (
                    self.model
                    .forward_visual_frontend(
                        video_chunk
                    )
                )

                # -------------------------------
                # Safety check
                # -------------------------------

                if (
                    audio_embed.shape[1]
                    != visual_embed.shape[1]
                ):

                    print(
                        "[TalkNet] Temporal mismatch:",
                        audio_embed.shape,
                        visual_embed.shape
                    )

                    continue

                # -------------------------------
                # Cross attention
                # -------------------------------

                (
                    audio_embed,
                    visual_embed
                ) = (
                    self.model
                    .forward_cross_attention(
                        audio_embed,
                        visual_embed
                    )
                )

                # -------------------------------
                # Audio-visual backend
                # -------------------------------

                output = (
                    self.model
                    .forward_audio_visual_backend(
                        audio_embed,
                        visual_embed
                    )
                )

                # -------------------------------
                # TalkNet classifier
                # 256 -> 2
                # -------------------------------

                if (
                    hasattr(self, "lossAV")
                    and self.lossAV is not None
                ):

                    logits = self.lossAV.FC(
                        output
                    )

                    probabilities = F.softmax(
                        logits,
                        dim=-1
                    )

                    speaking_scores = (
                        probabilities[:, 1]
                    )

                    scores.extend(
                        speaking_scores
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    logits_list.append(
                        logits
                        .detach()
                        .cpu()
                        .numpy()
                    )

                else:

                    print(
                        "[TalkNet] lossAV classifier unavailable."
                    )

                    return -10.0, None

            # -------------------------------
            # No scores
            # -------------------------------

            if len(scores) == 0:
                return -10.0, None

            scores_array = np.asarray(
                scores,
                dtype=np.float32
            )

            score = float(
                np.mean(scores_array)
            )

            if len(logits_list) > 0:

                output_array = np.concatenate(
                    logits_list,
                    axis=0
                )

            else:

                output_array = scores_array

            return (
                score,
                output_array
            )

        except Exception as exc:

            print(
                "[TalkNet] Inference error:",
                repr(exc)
            )

            return (
                -10.0,
                None
            )