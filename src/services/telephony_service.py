"""Telephony Service module for real-time automated voice calls and interactive call dialogs with farmers."""

import logging
import uuid
from typing import Any, Dict, Optional, Union
from src.agents.voice import VoiceAgent

logger = logging.getLogger(__name__)


class TelephonyService:
    """Service for handling automated outbound advisory voice calls and inbound phone calls with farmers."""

    def __init__(self, voice_agent: Optional[VoiceAgent] = None) -> None:
        """Initialize TelephonyService with VoiceAgent reference.

        Args:
            voice_agent: Optional VoiceAgent instance.
        """
        self.voice_agent = voice_agent or VoiceAgent()
        self.active_calls: Dict[str, Dict[str, Any]] = {}

    def trigger_outbound_call(
        self,
        farmer_phone: str,
        farmer_location: str = "Muzaffarnagar, Uttar Pradesh",
        commodity: str = "Wheat",
        quantity_quintals: float = 50.0,
        language: str = "hi-IN",
    ) -> Dict[str, Any]:
        """Trigger an automated outbound advisory voice call to a farmer.

        Args:
            farmer_phone: Farmer's phone number.
            farmer_location: Location or district of farmer.
            commodity: Crop name.
            quantity_quintals: Quantity in quintals.
            language: Preferred spoken language ('hi-IN' or 'en-IN').

        Returns:
            Dict containing call_id, status, spoken_script, audio_bytes, and market recommendation payload.
        """
        call_id = f"CALL_{uuid.uuid4().hex[:8].upper()}"

        if "hi" in language:
            query = f"{farmer_location} में {quantity_quintals:g} क्विंटल {commodity} के लिए मंडी भाव"
        else:
            query = f"Best mandi price for {quantity_quintals:g} quintals of {commodity} in {farmer_location}"

        voice_res = self.voice_agent.process_voice_query(
            query_input=query,
            input_type="text",
            language=language,
        )

        call_record = {
            "call_id": call_id,
            "phone": farmer_phone,
            "status": "COMPLETED",
            "direction": "OUTBOUND",
            "spoken_script": voice_res.get("response_text", ""),
            "audio_bytes": voice_res.get("audio_bytes", b""),
            "audio_mime_type": voice_res.get("mime_type", "audio/mp3"),
            "pipeline_data": voice_res.get("pipeline_data", {}),
        }

        self.active_calls[call_id] = call_record
        return call_record

    def handle_inbound_call(
        self,
        farmer_phone: str,
        speech_input: Optional[Union[str, bytes]] = None,
        language: str = "hi-IN",
    ) -> Dict[str, Any]:
        """Handle an inbound voice call turn from a farmer speaking into the phone.

        Args:
            farmer_phone: Caller phone number.
            speech_input: Spoken text query or recorded call audio bytes.
            language: Preferred language code ('hi-IN' or 'en-IN').

        Returns:
            Dict containing call_id, spoken_response, audio_bytes, and status.
        """
        call_id = f"CALL_IN_{uuid.uuid4().hex[:8].upper()}"

        input_type = "audio" if isinstance(speech_input, bytes) else "text"
        query_val = speech_input if speech_input else "मंडी भाव की जानकारी"

        voice_res = self.voice_agent.process_voice_query(
            query_input=query_val,
            input_type=input_type,
            language=language,
        )

        call_record = {
            "call_id": call_id,
            "phone": farmer_phone,
            "status": "COMPLETED",
            "direction": "INBOUND",
            "transcript": voice_res.get("transcript", ""),
            "spoken_response": voice_res.get("response_text", ""),
            "audio_bytes": voice_res.get("audio_bytes", b""),
            "audio_mime_type": voice_res.get("mime_type", "audio/mp3"),
            "intent": voice_res.get("intent", ""),
            "entities": voice_res.get("entities", {}),
        }

        self.active_calls[call_id] = call_record
        return call_record
