# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import logging

from verl.experimental.agent_loop.tool_parser import HermesToolParser


class DummyTokenizer:
    def __init__(self, text: str):
        self.text = text

    def decode(self, response_ids):
        return self.text


def test_hermes_parser_logs_raw_tool_request_on_decode_failure(caplog):
    malformed_tool_request = '{"name": "search_corpus", "arguments": {"command": "rg "bad" ."}}'
    text = f"<tool_call>\n{malformed_tool_request}\n</tool_call>"
    parser = HermesToolParser(DummyTokenizer(text))

    with caplog.at_level(logging.ERROR):
        content, function_calls = asyncio.run(parser.extract_tool_calls([1, 2, 3]))

    assert content.strip() == ""
    assert function_calls == []
    assert "Failed to decode tool call" in caplog.text
    assert "Raw tool request:" in caplog.text
    assert malformed_tool_request in caplog.text


def test_hermes_parser_recovers_python_literal_tool_call(caplog):
    python_literal_tool_request = (
        "{'name': 'search_corpus', "
        "'arguments': {'command': \"rg -n -m 3 'Loomis-Parry Residence' .\"}}"
    )
    text = f"<tool_call>\n{python_literal_tool_request}\n</tool_call>"
    parser = HermesToolParser(DummyTokenizer(text))

    with caplog.at_level(logging.WARNING):
        content, function_calls = asyncio.run(parser.extract_tool_calls([1, 2, 3]))

    assert content.strip() == ""
    assert len(function_calls) == 1
    assert function_calls[0].name == "search_corpus"
    assert function_calls[0].arguments == '{"command": "rg -n -m 3 \'Loomis-Parry Residence\' ."}'
    assert "Recovered non-JSON Python-literal tool call" in caplog.text
