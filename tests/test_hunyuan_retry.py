from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from asset_refiner.config import DEFAULT_CONFIG
from asset_refiner.exceptions import HunyuanApiError
from asset_refiner.hunyuan_backend import submit_job_with_retry


class FakeTencentClient:
    def __init__(self) -> None:
        self.call_count = 0

    def call(self, action: str, payload: dict[str, object]) -> dict[str, str]:
        self.call_count += 1
        if self.call_count == 1:
            raise HunyuanApiError(
                f"Tencent API {action} error FailedOperation.RequestTimeout: backend request timed out"
            )
        return {"JobId": "job-after-retry"}


class HunyuanSubmitRetryTests(unittest.TestCase):
    @patch("asset_refiner.hunyuan_backend.time.sleep")
    def test_request_timeout_is_retried(self, sleep_mock) -> None:
        client = FakeTencentClient()
        config = copy.deepcopy(DEFAULT_CONFIG["hunyuan"])
        config["submit_retry_interval_seconds"] = 0

        response = submit_job_with_retry(
            client,
            submit_action="SubmitReduceFaceJob",
            submit_payload={"File3D": {}},
            config=config,
        )

        self.assertEqual(response["JobId"], "job-after-retry")
        self.assertEqual(client.call_count, 2)
        sleep_mock.assert_called_once_with(0.0)


if __name__ == "__main__":
    unittest.main()
