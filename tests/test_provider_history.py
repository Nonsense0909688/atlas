from atlas.providers.base import BaseProvider, ChatResponse
from atlas.models import ModelConfig


class RecordingProvider(BaseProvider):
    name = "recording"

    def __init__(self):
        self.last_input = None

    def chat(self, model, input, tools=None, stream=False):
        self.last_input = input
        return ChatResponse(text="ok")


def test_default_chat_messages_keeps_all_turns():
    provider = RecordingProvider()
    model = ModelConfig(id="test", provider="recording")

    provider.chat_messages(model, [
        {"role": "user", "content": "Remember the code word is amber."},
        {"role": "assistant", "content": "I will remember."},
        {"role": "user", "content": "What is the code word?"},
    ])

    assert "Remember the code word is amber." in provider.last_input
    assert "I will remember." in provider.last_input
    assert "What is the code word?" in provider.last_input
