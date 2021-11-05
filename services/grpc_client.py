# Copyright 2018 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""gRPC Python helloworld.Greeter client with channel options and call timeout parameters."""

from __future__ import print_function
import os
import json
import pickle
from typing import List

import grpc
from grpc._channel import _InactiveRpcError

from services import recognizers_pb2
from services import recognizers_pb2_grpc

# Text to test service availability
TEST_TEXT = "I love you 3000 times"
RECOGNIZER_TARGET = "localhost:50051"


def run_recognizer(text: str) -> list:
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.

    # Get Recognizers Text service target from env first and config file next
    if os.getenv('RECOGNIZER_TARGET'):
        target = os.getenv('RECOGNIZER_TARGET')
    else:
        target = RECOGNIZER_TARGET
    # For more channel options, please see https://grpc.io/grpc/core/group__grpc__arg__keys.html
    with grpc.insecure_channel(target=target,
                               options=[('grpc.lb_policy_name', 'pick_first'),
                                        ('grpc.enable_retries', 0),
                                        ('grpc.keepalive_timeout_ms', 10000)
                                        ]) as channel:
        stub = recognizers_pb2_grpc.RecognizersTextStub(channel)
        # Timeout in seconds.
        # Please refer gRPC Python documents for more detail. https://grpc.io/grpc/python/grpc.html
        response = stub.GetRecognizeResult(recognizers_pb2.TextRequest(text=text), timeout=60)
        result = json.loads(response.result)
        return result


def run_test():
    """
    Test services
    """
    # Test grpc services
    test_sentence = "I love you 3000 times"
    recognizers_result = run_recognizer(test_sentence)
    print(recognizers_result)


if __name__ == '__main__':
    run_test()
