"""WhatsApp Service module for handling WhatsApp chatbot messages and voice notes in real-time."""

import logging
from typing import Any, Dict, Optional, Union
from src.agents.voice import VoiceAgent

logger = logging.getLogger(__name__)


class WhatsAppService:
    """Service for handling WhatsApp chatbot text and voice note interactions with farmers."""

    def __init__(self, voice_agent: Optional[VoiceAgent] = None) -> None:
        """Initialize WhatsAppService with VoiceAgent reference.

        Args:
            voice_agent: Optional VoiceAgent instance.
        """
        self.voice_agent = voice_agent or VoiceAgent()

    def process_incoming_message(
        self,
        from_phone: str,
        message_body: Optional[str] = None,
        media_url: Optional[str] = None,
        media_bytes: Optional[bytes] = None,
        media_type: Optional[str] = None,
        language: str = "hi-IN",
    ) -> Dict[str, Any]:
        """Process an incoming WhatsApp message or voice note from a farmer.

        Args:
            from_phone: Farmer's phone number (e.g. '+919876543210').
            message_body: Optional text message body.
            media_url: Optional media URL (for voice notes/audio).
            media_bytes: Optional raw audio bytes for voice note.
            media_type: Optional mime type (e.g. 'audio/ogg', 'audio/wav', 'audio/mp3').
            language: Preferred language code ('hi-IN' or 'en-IN').

        Returns:
            Dict containing formatted WhatsApp reply text, audio_bytes, recipient, and status.
        """
        is_voice_note = bool(media_bytes or media_url or (media_type and "audio" in media_type))

        if is_voice_note and media_bytes:
            voice_res = self.voice_agent.process_voice_query(
                query_input=media_bytes,
                input_type="audio",
                language=language,
            )
        elif message_body and message_body.strip():
            voice_res = self.voice_agent.process_voice_query(
                query_input=message_body.strip(),
                input_type="text",
                language=language,
            )
        else:
            voice_res = self.voice_agent.process_voice_query(
                query_input="मुजफ्फरनगर में 50 क्विंटल गेहूं के लिए मंडी भाव",
                input_type="text",
                language=language,
            )

        raw_text = voice_res.get("response_text", "")
        entities = voice_res.get("entities", {})
        pipeline_data = voice_res.get("pipeline_data", {})
        winner = pipeline_data.get("winner")

        if winner:
            wa_reply = (
                f"🌾 *K.I.S.A.N. AI WhatsApp Assistant* 🌾\n\n"
                f"{raw_text}\n\n"
                f"📊 *Live Details*:\n"
                f"• *Crop*: {entities.get('commodity')}\n"
                f"• *Quantity*: {entities.get('quantity_quintals')} Quintals\n"
                f"• *Recommended Mandi*: {winner.get('market_name')}\n"
                f"• *Expected Modal Price*: ₹{winner.get('expected_modal_price_inr', 0):,.0f}/quintal\n"
                f"• *Net Profit*: ₹{winner.get('net_expected_profit_inr', 0):,.0f}\n\n"
                f"💬 _Reply to this message or send a voice note for more queries!_"
            )
        else:
            wa_reply = f"🌾 *K.I.S.A.N. AI WhatsApp Assistant* 🌾\n\n{raw_text}"

        return {
            "status": "success",
            "recipient_phone": from_phone,
            "reply_text": wa_reply,
            "audio_bytes": voice_res.get("audio_bytes", b""),
            "audio_mime_type": voice_res.get("mime_type", "audio/mp3"),
            "transcript": voice_res.get("transcript", ""),
            "intent": voice_res.get("intent", ""),
            "entities": entities,
        }

    def handle_webhook_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse standard WhatsApp Webhook (e.g. Twilio / WhatsApp Business API) payload.

        Args:
            payload: Webhook POST body payload dictionary.

        Returns:
            Processed response payload dictionary.
        """
        from_phone = payload.get("From", payload.get("from", "+919000000000"))
        body = payload.get("Body", payload.get("body", ""))
        media_url = payload.get("MediaUrl0", payload.get("media_url"))
        media_type = payload.get("MediaContentType0", payload.get("media_type"))

        return self.process_incoming_message(
            from_phone=from_phone,
            message_body=body,
            media_url=media_url,
            media_type=media_type,
        )
