import asyncio
from typing import AsyncIterable, Annotated
from decouple import config
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastui import prebuilt_html, FastUI, AnyComponent
from fastui import components as c
from fastui.components.display import DisplayLookup, DisplayMode
from fastui.events import PageEvent, GoToEvent
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
import requests

# Create the app object
app = FastAPI()

# Message history
app.message_history = []

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {config('HUGGINGFACE_API_KEY')}"}

# Message history model
class MessageHistoryModel(BaseModel):
    message: str = Field(title='Message')

# Chat form
class ChatForm(BaseModel):
    chat: str = Field(title=' ', max_length=1000)

# Root endpoint
@app.get('/api/', response_model=FastUI, response_model_exclude_none=True)
def api_index(chat: str | None = None, reset: bool = False) -> list[AnyComponent]:
    if reset:
        app.message_history = []
    return [
        c.PageTitle(text='FastUI Chatbot'),
        c.Page(
            components=[
                # Header
                c.Heading(text='FastUI Chatbot'),
                c.Paragraph(text='This is a simple chatbot built with FastUI and Mistral-7B-Instruct-v0.2.'),
                # Chat history
                c.Table(
                    data=app.message_history,
                    data_model=MessageHistoryModel,
                    columns=[DisplayLookup(field='message', mode=DisplayMode.markdown, table_width_percent=100)],
                    no_data_message='No messages yet.',
                ),
                # Chat form
                c.ModelForm(model=ChatForm, submit_url=".", method='GOTO'),
                # Reset chat
                c.Link(
                    components=[c.Text(text='Reset Chat')],
                    on_click=GoToEvent(url='/?reset=true'),
                ),
                # Chatbot response
                c.Div(
                    components=[
                        c.ServerLoad(
                            path=f"/sse/{chat}",
                            sse=True,
                            load_trigger=PageEvent(name='load'),
                            components=[],
                        )
                    ],
                    class_name='my-2 p-2 border rounded'),
            ],
        ),
        # Footer
        c.Footer(
            extra_text='Made with Love',
            links=[]
        )
    ]


# SSE endpoint
@app.get('/api/sse/{prompt}')
async def sse_ai_response(prompt: str) -> StreamingResponse:
    # Check if prompt is empty
    if prompt is None or prompt == '' or prompt == 'None':
        return StreamingResponse(empty_response(), media_type='text/event-stream')
    return StreamingResponse(ai_response_generator(prompt), media_type='text/event-stream')

# Empty response generator
async def empty_response() -> AsyncIterable[str]:
    # Send the message
    m = FastUI(root=[c.Markdown(text='')])
    msg = f'data: {m.model_dump_json(by_alias=True, exclude_none=True)}\n\n'
    yield msg
    # Avoid the browser reconnecting
    while True:
        yield msg
        await asyncio.sleep(10)

# Hugging Face response generator
async def ai_response_generator(prompt: str) -> AsyncIterable[str]:
    # Output variables
    output = f"**User:** {prompt}\n\n"
    msg = ''
    # Prompt template for message history
    prompt_template = "Previous messages:\n"
    for message_history in app.message_history:
        prompt_template += message_history.message + "\n"
    prompt_template += f"Human: {prompt}"
    # Hugging Face chat messages
    huggingface_messages = {
        "inputs": prompt_template
    }
    # Stream the chat
    output += f"**Chatbot:** "
    response = requests.post(API_URL, headers=headers, json=huggingface_messages)
    for chunk in response.json():
        if token := chunk['generated_text']:
            # Add the token to the output
            output += token
            # Send the message
            m = FastUI(root=[c.Markdown(text=output)])
            msg = f'data: {m.model_dump_json(by_alias=True, exclude_none=True)}\n\n'
            yield msg
    # Append the message to the history
    message = MessageHistoryModel(message=output)
    app.message_history.append(message)
    # Avoid the browser reconnecting
    while True:
        yield msg
        await asyncio.sleep(10)

# Prebuilt HTML
@app.get('/{path:path}')
async def html_landing() -> HTMLResponse:
    """Simple HTML page which serves the React app, comes last as it matches all paths."""
    return HTMLResponse(prebuilt_html(title='FastUI Demo'))