from pathlib import Path
from typing import Dict, Optional
import string
import random

import requests
import textract


def gen_random_string(
    choices: str = string.ascii_lowercase,
    length: int = 10
) -> str:
    """Generate a random file name to avoid collisions.

    Args:
        choices (str, optional): characters to choose from. Defaults to ``string.ascii_lowercase``.
        length (int, optional): length of the random string. Defaults to 10.
    
    Returns:
        str: random file name.
    """
    return "".join(random.choices(choices, k=length))


class RemoteFile:
    """Class for managing files at a remote URL.

    Methods:
        delete: delete the file.
        download: download the file.
        extract_text: extract text from the file.
    
    Attributes:
        url: URL of the remote file.
        extension: file extension.
        name: file name.
        stem: file name without the extension.
    """
    def __init__(
        self,
        url: str
    ) -> None:
        """Create a new RemoteFile instance.

        Args:
            url (str): URL of the remote file.
        """
        self.url = url
        self._local_path = Path(f"{gen_random_string()}{self.extension}")
        self._is_deleted = False
        self._is_downloaded = False

    def delete(self):
        self._local_path.unlink()
        self._is_deleted = True

    def download(self, headers: Optional[Dict[str, str]] = None) -> None:
        """Download the file at the given URL.

        Args:
            headers (Optional[Dict[str, str]], optional): HTTP headers to send with the request. Defaults to None.
        """
        # generate a random file name to avoid collisions
        response = requests.get(self.url, headers=headers)
        response.raise_for_status()
        with open(self._local_path, "wb") as f:
            f.write(response.content)
        
        self._is_downloaded = True
    
    @property
    def _url_path(self) -> Path:
        return Path(self.url)
    
    @property
    def extension(self) -> str:
        return self._url_path.suffix
    
    @property
    def name(self) -> str:
        return self._url_path.name

    @property
    def stem(self) -> str:
        return self._url_path.stem
    
    def extract_text(self) -> str:
        """Extract text from the file.

        Returns:
            str: text content of the file.
        """
        return textract.process(str(self._local_path)).decode()