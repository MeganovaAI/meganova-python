from pathlib import Path
from typing import BinaryIO, Tuple, Union
from ..transport import SyncTransport
from ..models.audio import TranscriptionResponse


class AudioResource:
    def __init__(self, transport: SyncTransport):
        self._transport = transport

    def transcribe(
        self,
        file: Union[str, Path, BinaryIO],
        *,
        model: str = "Systran/faster-whisper-large-v3",
        filename: str = "audio.mp3",
    ) -> TranscriptionResponse:
        """Transcribe audio to text.

        Args:
            file: Path to an audio file, or a file-like object.
            model: Transcription model name.
            filename: Filename hint when passing a file-like object.

        Returns:
            TranscriptionResponse containing the transcribed text.
        """
        if isinstance(file, (str, Path)):
            path = Path(file)
            with open(path, "rb") as f:
                files = {
                    "file": (path.name, f, "audio/mpeg"),
                }
                data = {"model": model}
                result = self._transport.request(
                    "POST", "/audio/transcriptions", files=files, data=data,
                )
        else:
            files = {
                "file": (filename, file, "audio/mpeg"),
            }
            data = {"model": model}
            result = self._transport.request(
                "POST", "/audio/transcriptions", files=files, data=data,
            )
        return TranscriptionResponse(**result)
