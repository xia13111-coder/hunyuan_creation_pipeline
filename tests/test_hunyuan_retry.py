from __future__ import annotations

import copy
import io
import ssl
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from asset_refiner.config import DEFAULT_CONFIG
from asset_refiner.exceptions import HunyuanApiError
from asset_refiner.hunyuan_backend import (
    download_url,
    run_job,
    submit_job_with_retry,
    submit_reduce_face_with_temporary_upload,
)


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

    @patch("asset_refiner.hunyuan_backend.time.sleep")
    @patch("asset_refiner.hunyuan_backend.submit_reduce_face")
    @patch("asset_refiner.hunyuan_backend.upload_to_temporary_host")
    def test_temporary_upload_http_error_is_retried(
        self,
        upload_mock,
        submit_mock,
        sleep_mock,
    ) -> None:
        upload_mock.side_effect = [
            HunyuanApiError("Temporary upload HTTP 502: Bad Gateway"),
            "https://temporary.example/model.glb",
        ]
        submit_mock.return_value = {"job_id": "job-after-upload-retry"}
        config = copy.deepcopy(DEFAULT_CONFIG)

        result = submit_reduce_face_with_temporary_upload(
            FakeTencentClient(),
            local_input=Path("model.glb"),
            input_type="GLB",
            config=config,
        )

        self.assertEqual(result["job_id"], "job-after-upload-retry")
        self.assertEqual(upload_mock.call_count, 2)
        submit_mock.assert_called_once()
        sleep_mock.assert_called_once_with(10.0)

    @patch("asset_refiner.hunyuan_backend.time.sleep")
    @patch("asset_refiner.hunyuan_backend.submit_reduce_face")
    @patch("asset_refiner.hunyuan_backend.upload_to_temporary_host")
    def test_tencent_download_error_reuploads_temporary_file(
        self,
        upload_mock,
        submit_mock,
        sleep_mock,
    ) -> None:
        upload_mock.side_effect = [
            "https://temporary.example/first.glb",
            "https://temporary.example/second.glb",
        ]
        submit_mock.side_effect = [
            HunyuanApiError("FailedOperation.DownloadError: download failed"),
            {"job_id": "job-after-download-retry"},
        ]
        config = copy.deepcopy(DEFAULT_CONFIG)

        result = submit_reduce_face_with_temporary_upload(
            FakeTencentClient(),
            local_input=Path("model.glb"),
            input_type="GLB",
            config=config,
        )

        self.assertEqual(result["job_id"], "job-after-download-retry")
        self.assertEqual(upload_mock.call_count, 2)
        self.assertEqual(submit_mock.call_count, 2)
        sleep_mock.assert_called_once_with(10.0)

    @patch("asset_refiner.hunyuan_backend.time.sleep")
    @patch("asset_refiner.hunyuan_backend.submit_reduce_face")
    @patch("asset_refiner.hunyuan_backend.upload_to_temporary_host")
    def test_inner_error_reuploads_and_submits_a_new_job(
        self,
        upload_mock,
        submit_mock,
        sleep_mock,
    ) -> None:
        upload_mock.side_effect = [
            "https://temporary.example/first.glb",
            "https://temporary.example/second.glb",
        ]
        submit_mock.side_effect = [
            HunyuanApiError(
                "Hunyuan reduce_face job failed: "
                "FailedOperation.InnerError 服务内部错误，请重试。"
            ),
            {"job_id": "new-job-after-inner-error"},
        ]
        config = copy.deepcopy(DEFAULT_CONFIG)

        result = submit_reduce_face_with_temporary_upload(
            FakeTencentClient(),
            local_input=Path("model.glb"),
            input_type="GLB",
            config=config,
        )

        self.assertEqual(result["job_id"], "new-job-after-inner-error")
        self.assertEqual(upload_mock.call_count, 2)
        self.assertEqual(submit_mock.call_count, 2)
        self.assertEqual(
            [item.args[1].url for item in submit_mock.call_args_list],
            [
                "https://temporary.example/first.glb",
                "https://temporary.example/second.glb",
            ],
        )
        sleep_mock.assert_called_once_with(30.0)

    @patch("asset_refiner.hunyuan_backend.time.sleep")
    @patch("asset_refiner.hunyuan_backend.submit_reduce_face")
    @patch("asset_refiner.hunyuan_backend.upload_to_temporary_host")
    def test_inner_error_stops_after_configured_job_attempts(
        self,
        upload_mock,
        submit_mock,
        sleep_mock,
    ) -> None:
        upload_mock.side_effect = [
            "https://temporary.example/first.glb",
            "https://temporary.example/second.glb",
        ]
        error = HunyuanApiError(
            "Hunyuan reduce_face job failed: "
            "FailedOperation.InnerError 服务内部错误，请重试。"
        )
        submit_mock.side_effect = [error, error]
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["hunyuan"]["job_failure_retry"]["max_attempts"] = 2
        config["hunyuan"]["job_failure_retry"]["retry_interval_seconds"] = 0

        with self.assertRaisesRegex(
            HunyuanApiError,
            "FailedOperation.InnerError",
        ):
            submit_reduce_face_with_temporary_upload(
                FakeTencentClient(),
                local_input=Path("model.glb"),
                input_type="GLB",
                config=config,
            )

        self.assertEqual(upload_mock.call_count, 2)
        self.assertEqual(submit_mock.call_count, 2)
        sleep_mock.assert_called_once_with(0.0)

    @patch("asset_refiner.hunyuan_backend.time.sleep")
    @patch("asset_refiner.hunyuan_backend.submit_reduce_face")
    @patch("asset_refiner.hunyuan_backend.upload_to_temporary_host")
    def test_nonretryable_failed_job_is_not_resubmitted(
        self,
        upload_mock,
        submit_mock,
        sleep_mock,
    ) -> None:
        upload_mock.return_value = "https://temporary.example/model.glb"
        submit_mock.side_effect = HunyuanApiError(
            "Hunyuan reduce_face job failed: "
            "FailedOperation.InvalidParameter invalid model"
        )
        config = copy.deepcopy(DEFAULT_CONFIG)

        with self.assertRaisesRegex(HunyuanApiError, "InvalidParameter"):
            submit_reduce_face_with_temporary_upload(
                FakeTencentClient(),
                local_input=Path("model.glb"),
                input_type="GLB",
                config=config,
            )

        upload_mock.assert_called_once()
        submit_mock.assert_called_once()
        sleep_mock.assert_not_called()

    @patch("asset_refiner.hunyuan_backend.time.sleep")
    @patch("asset_refiner.hunyuan_backend.urllib.request.urlopen")
    def test_result_download_ssl_eof_is_retried(
        self,
        urlopen_mock,
        sleep_mock,
    ) -> None:
        response = io.BytesIO(b"GLB!")
        response.headers = {"Content-Length": "4"}
        urlopen_mock.side_effect = [
            urllib.error.URLError(
                ssl.SSLEOFError(8, "EOF occurred in violation of protocol")
            ),
            response,
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "target.glb"
            result = download_url(
                "https://example.invalid/target.glb",
                destination,
                max_attempts=3,
                retry_interval_seconds=2,
                retry_backoff_factor=2,
                timeout_seconds=7,
            )

            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), b"GLB!")
            self.assertFalse(destination.with_name("target.glb.part").exists())

        self.assertEqual(urlopen_mock.call_count, 2)
        urlopen_mock.assert_called_with(
            "https://example.invalid/target.glb",
            timeout=7,
        )
        sleep_mock.assert_called_once_with(2.0)

    @patch("asset_refiner.hunyuan_backend.time.sleep")
    @patch("asset_refiner.hunyuan_backend.urllib.request.urlopen")
    def test_result_download_http_403_is_not_retried(
        self,
        urlopen_mock,
        sleep_mock,
    ) -> None:
        urlopen_mock.side_effect = urllib.error.HTTPError(
            "https://example.invalid/target.glb",
            403,
            "Forbidden",
            {},
            io.BytesIO(b"forbidden"),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "target.glb"
            with self.assertRaises(urllib.error.HTTPError):
                download_url(
                    "https://example.invalid/target.glb",
                    destination,
                    max_attempts=5,
                )

            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name("target.glb.part").exists())

        urlopen_mock.assert_called_once()
        sleep_mock.assert_not_called()

    @patch("asset_refiner.hunyuan_backend.time.sleep")
    def test_describe_ssl_eof_is_retried_without_resubmitting(
        self,
        sleep_mock,
    ) -> None:
        network_error = urllib.error.URLError(
            ssl.SSLEOFError(8, "EOF occurred in violation of protocol")
        )
        describe_error = HunyuanApiError(
            "Tencent API request failed for DescribeReduceFaceJob"
        )
        describe_error.__cause__ = network_error

        client = Mock()
        client.call.side_effect = [
            {"JobId": "existing-job"},
            describe_error,
            {
                "Status": "DONE",
                "ResultFile3Ds": [
                    {"Type": "GLB", "Url": "https://example.invalid/result.glb"}
                ],
            },
        ]
        config = copy.deepcopy(DEFAULT_CONFIG["hunyuan"])
        config["submit_max_retries"] = 1
        config["describe_retry_interval_seconds"] = 0

        result = run_job(
            client,
            submit_action="SubmitReduceFaceJob",
            describe_action="DescribeReduceFaceJob",
            submit_payload={
                "File3D": {},
                "__hunyuan_retry_config": config,
            },
            poll_interval_seconds=0,
            timeout_seconds=30,
            stage_name="reduce_face",
        )

        self.assertEqual(result["job_id"], "existing-job")
        self.assertEqual(client.call.call_count, 3)
        self.assertEqual(
            [item.args[0] for item in client.call.call_args_list],
            [
                "SubmitReduceFaceJob",
                "DescribeReduceFaceJob",
                "DescribeReduceFaceJob",
            ],
        )
        sleep_mock.assert_called_once_with(0.0)

    @patch("asset_refiner.hunyuan_backend.time.sleep")
    def test_describe_permanent_api_error_is_not_retried(
        self,
        sleep_mock,
    ) -> None:
        client = Mock()
        client.call.side_effect = [
            {"JobId": "existing-job"},
            HunyuanApiError(
                "Tencent API DescribeReduceFaceJob error "
                "AuthFailure.SignatureFailure: invalid signature"
            ),
        ]
        config = copy.deepcopy(DEFAULT_CONFIG["hunyuan"])
        config["submit_max_retries"] = 1

        with self.assertRaisesRegex(HunyuanApiError, "SignatureFailure"):
            run_job(
                client,
                submit_action="SubmitReduceFaceJob",
                describe_action="DescribeReduceFaceJob",
                submit_payload={
                    "File3D": {},
                    "__hunyuan_retry_config": config,
                },
                poll_interval_seconds=0,
                timeout_seconds=30,
                stage_name="reduce_face",
            )

        self.assertEqual(client.call.call_count, 2)
        sleep_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
