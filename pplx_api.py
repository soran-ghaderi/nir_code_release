import anthropic
import os

# Set your API key as an environment variable
os.environ["ANTHROPIC_API_KEY"] = (
    "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"
)

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1000,
    temperature=0,
    system="You are a world-class poet. Respond only with short poems.",
    messages=[
        {
            "role": "user",
            "content": [{"type": "text", "text": "Why is the ocean salty?"}],
        }
    ],
)
print(message.content)
