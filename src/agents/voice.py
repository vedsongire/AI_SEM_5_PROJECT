"""Voice Agent module for real-time speech recognition, multilingual vernacular NLU, and dynamic audio synthesis.

Conforms to K.I.S.A.N. AI principles:
- 100% data-driven: queries local fallback CSVs and active multi-agent pipeline.
- No hardcoded data: extracts crops, quantities, and locations dynamically.
- Vernacular Hindi & Indian English support for text and voice audio.
"""

import base64
import io
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False

from src.agents.scout import ScoutAgent
from src.agents.predictor import PredictorAgent
from src.agents.planner import PlannerAgent


class VoiceAgent:
    """Agent responsible for speech-to-text, vernacular NLU parsing, multi-agent pipeline orchestration,
    and voice audio synthesis for farmer interactions.
    """

    # Vernacular Hindi and English crop mapping to canonical dataset names
    CROP_SYNONYMS = {
        "wheat": "Wheat",
        "gehun": "Wheat",
        "gehu": "Wheat",
        "गेहूं": "Wheat",
        "गेहू": "Wheat",
        "गहू": "Wheat",
        "potato": "Potato",
        "aalu": "Potato",
        "aaloo": "Potato",
        "आलू": "Potato",
        "आलु": "Potato",
        "batata": "Potato",
        "onion": "Onion",
        "pyaj": "Onion",
        "pyaz": "Onion",
        "kanda": "Onion",
        "प्याज": "Onion",
        "कांदा": "Onion",
        "tomato": "Tomato",
        "tamatar": "Tomato",
        "टमाटर": "Tomato",
        "rice": "Rice",
        "chawal": "Rice",
        "dhan": "Paddy(Dhan)(Common)",
        "paddy": "Paddy(Dhan)(Common)",
        "चावल": "Rice",
        "धान": "Paddy(Dhan)(Common)",
        "cotton": "Cotton",
        "kapas": "Cotton",
        "कपास": "Cotton",
        "रूई": "Cotton",
        "mustard": "Mustard",
        "sarson": "Mustard",
        "सरसों": "Mustard",
        "राई": "Mustard",
        "soyabean": "Soyabean",
        "soybean": "Soyabean",
        "सोयाबीन": "Soyabean",
        "maize": "Maize",
        "makka": "Maize",
        "corn": "Maize",
        "मक्का": "Maize",
        "bhutta": "Maize",
        "banana": "Banana",
        "kela": "Banana",
        "केला": "Banana",
        "apple": "Apple",
        "seb": "Apple",
        "सेब": "Apple",
        "gram": "Bengal Gram(Gram)(Whole)",
        "chana": "Bengal Gram(Gram)(Whole)",
        "चना": "Bengal Gram(Gram)(Whole)",
        "bengal gram": "Bengal Gram(Gram)(Whole)",
        "bajra": "Bajra(Pearl Millet-Cumbu)",
        "बाजरा": "Bajra(Pearl Millet-Cumbu)",
        "carrot": "Carrot",
        "gajar": "Carrot",
        "गाजर": "Carrot",
        "cauliflower": "Cauliflower",
        "phool gobhi": "Cauliflower",
        "गोभी": "Cauliflower",
        "फूलगोभी": "Cauliflower",
        "cabbage": "Cabbage",
        "bandha gobhi": "Cabbage",
        "पत्तागोभी": "Cabbage",
        "brinjal": "Brinjal",
        "baingan": "Brinjal",
        "बैंगन": "Brinjal",
        "green chilli": "Green Chilli",
        "mirch": "Green Chilli",
        "हरी मिर्च": "Green Chilli",
        "मिर्च": "Green Chilli",
        "bhindi": "Bhindi(Ladies Finger)",
        "okra": "Bhindi(Ladies Finger)",
        "ladies finger": "Bhindi(Ladies Finger)",
        "भिंडी": "Bhindi(Ladies Finger)",
        "sugarcane": "Sugarcane",
        "ganna": "Sugarcane",
        "गन्ना": "Sugarcane",
    }

    # Hindi number words mapping
    HINDI_NUMBERS = {
        "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5,
        "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
        "पंद्रह": 15, "बीस": 20, "पच्चीस": 25, "तीस": 30, "पैंतीस": 35,
        "चालीस": 40, "पचास": 50, "साठ": 60, "सत्तर": 70, "अस्सी": 80,
        "नब्बे": 90, "सौ": 100, "डेढ़ सौ": 150, "दो सौ": 200, "ढाई सौ": 250,
        "तीन सौ": 300, "चार सौ": 400, "पांच सौ": 500, "हजार": 1000,
    }

    def __init__(
        self,
        scout_agent: Optional[ScoutAgent] = None,
        predictor_agent: Optional[PredictorAgent] = None,
        planner_agent: Optional[PlannerAgent] = None,
    ) -> None:
        """Initialize VoiceAgent with references to existing multi-agent systems."""
        self.scout = scout_agent or ScoutAgent()
        self.predictor = predictor_agent or PredictorAgent()
        self.planner = planner_agent or PlannerAgent()
        self._load_available_crops()

    def _load_available_crops(self) -> None:
        """Dynamically detect available crop names from data/ directory."""
        self.available_crops: List[str] = []
        project_root = Path(__file__).resolve().parent.parent.parent
        veg_dir = project_root / "data" / "data_vegetable_wise"
        if veg_dir.exists():
            for f in veg_dir.glob("*.csv"):
                crop_clean = f.stem.split("(")[0].strip()
                if crop_clean and crop_clean not in self.available_crops:
                    self.available_crops.append(crop_clean)

        # Fallback to standard crops if folder was empty
        if not self.available_crops:
            self.available_crops = ["Wheat", "Potato", "Onion", "Tomato", "Rice", "Cotton", "Mustard", "Soyabean"]

    def transcribe_audio(
        self,
        audio_source: Union[str, bytes, Path],
        language: str = "hi-IN",
    ) -> Dict[str, Any]:
        """Transcribe speech audio into text using real-time SpeechRecognition engine.

        Args:
            audio_source: Audio file path (str/Path) or raw audio bytes (e.g. WAV/AIFF).
            language: Speech recognition language code ('hi-IN' or 'en-IN').

        Returns:
            Dict containing status, transcript, and language code.
        """
        if not HAS_SPEECH_RECOGNITION:
            return {
                "status": "error",
                "message": "speech_recognition package not installed.",
                "transcript": "",
                "language": language,
            }

        recognizer: Any = sr.Recognizer()

        try:
            if isinstance(audio_source, (str, Path)):
                with sr.AudioFile(str(audio_source)) as source:
                    audio_data = recognizer.record(source)
            elif isinstance(audio_source, bytes):
                buffer = io.BytesIO(audio_source)
                with sr.AudioFile(buffer) as source:
                    audio_data = recognizer.record(source)
            else:
                return {
                    "status": "error",
                    "message": "Invalid audio source type.",
                    "transcript": "",
                    "language": language,
                }

            # In SpeechRecognition 3.14+, recognize_google is dynamically attached to Recognizer
            # at module load time (Recognizer.recognize_google = google.recognize_legacy), which causes
            # static type analyzers (Pylance/Pyright) to flag it as missing from the class definition.
            recognize_fn = getattr(recognizer, "recognize_google", None)
            if callable(recognize_fn):
                transcript = recognize_fn(audio_data, language=language)
            else:
                from speech_recognition.recognizers import google
                transcript = google.recognize_legacy(recognizer, audio_data, language=language)

            return {
                "status": "success",
                "transcript": transcript,
                "language": language,
            }
        except sr.UnknownValueError:
            return {
                "status": "error",
                "message": "Could not understand speech audio. Please speak clearly.",
                "transcript": "",
                "language": language,
            }
        except sr.RequestError as exc:
            return {
                "status": "error",
                "message": f"Speech recognition network service error: {exc}",
                "transcript": "",
                "language": language,
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Speech processing error: {exc}",
                "transcript": "",
                "language": language,
            }

    def _extract_commodity(self, text: str) -> Optional[str]:
        """Dynamically extract commodity name from query string."""
        clean_text = text.lower()

        # 1. Match against known synonyms (Hindi & English)
        for synonym, canonical in self.CROP_SYNONYMS.items():
            pattern = r"(?<!\w)" + re.escape(synonym) + r"(?!\w)"
            if re.search(pattern, clean_text, re.IGNORECASE):
                return canonical

        # 2. Match against discovered crops from data directory
        for crop in self.available_crops:
            if re.search(r"(?<!\w)" + re.escape(crop.lower()) + r"(?!\w)", clean_text):
                return crop

        return None

    def _extract_quantity(self, text: str) -> Optional[float]:
        """Dynamically extract harvest yield quantity in quintals."""
        clean_text = text.lower()

        # 1. Regex patterns for explicit units: quintal, ton, bag, kg
        # Quintals
        qtl_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:quintals?|qtl|qt|क्विंटल|किवंटल|क्विटल|कुंतल)",
            clean_text,
            re.IGNORECASE,
        )
        if qtl_match:
            try:
                return float(qtl_match.group(1))
            except ValueError:
                pass

        # Tons / Tonnes (1 Ton = 10 Quintals)
        ton_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:tons?|tonnes?|टन)",
            clean_text,
            re.IGNORECASE,
        )
        if ton_match:
            try:
                return round(float(ton_match.group(1)) * 10.0, 2)
            except ValueError:
                pass

        # Bags / Bori (Standard agricultural 50kg bag = 0.5 Quintal)
        bag_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:bags?|bori|बोरी|कट्टे|कट्टा|बैग)",
            clean_text,
            re.IGNORECASE,
        )
        if bag_match:
            try:
                return round(float(bag_match.group(1)) * 0.5, 2)
            except ValueError:
                pass

        # Kilograms / KG (100 KG = 1 Quintal)
        kg_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilo|kilograms?|किलो|किग्रा)",
            clean_text,
            re.IGNORECASE,
        )
        if kg_match:
            try:
                return round(float(kg_match.group(1)) / 100.0, 2)
            except ValueError:
                pass

        # 2. Check Hindi written number words preceding quintal
        for word, val in self.HINDI_NUMBERS.items():
            if f"{word} क्विंटल" in clean_text or f"{word} किवंटल" in clean_text:
                return float(val)

        # 3. Isolated realistic numbers (e.g. "50", "80", "100")
        isolated = re.findall(r"\b(\d{1,4}(?:\.\d+)?)\b", clean_text)
        for num_str in isolated:
            try:
                val = float(num_str)
                # Ignore pin codes (6 digits) or phone numbers (>10000) or very small numbers
                if 2 <= val <= 2000:
                    return val
            except ValueError:
                continue

        return None

    def _extract_location(self, text: str, commodity: Optional[str] = None, quantity: Optional[float] = None) -> Optional[str]:
        """Dynamically extract location / district name by stripping stop words and known entities."""
        clean = text

        # Strip crop names
        if commodity:
            for syn, can in self.CROP_SYNONYMS.items():
                if can == commodity:
                    clean = re.sub(re.escape(syn), " ", clean, flags=re.IGNORECASE)
            clean = re.sub(re.escape(commodity), " ", clean, flags=re.IGNORECASE)

        # Strip quantity numbers and units
        clean = re.sub(r"\d+(?:\.\d+)?\s*(?:quintals?|qtl|qt|tons?|kg|किलो|क्विंटल|बोरी|बैग)?", " ", clean, flags=re.IGNORECASE)

        # Strip common agricultural question filler words in Hindi and English
        stop_words = {
            "what", "is", "the", "best", "mandi", "market", "for", "price", "rate", "in", "near", "at", "to",
            "sell", "selling", "profit", "transport", "freight", "truck", "distance", "recommendation",
            "please", "tell", "me", "show", "give", "of", "and", "from", "i", "want", "have", "my", "which",
            "where", "can", "how", "much", "good", "better", "produce", "crop", "harvest",
            "नमस्ते", "किसान", "भाई", "का", "की", "के", "लिए", "सबसे", "अच्छी", "मंडी", "भाव", "क्या", "है",
            "बताओ", "बताइए", "चाहिए", "बेचना", "में", "से", "यहाँ", "नजदीकी", "लाभ", "खर्च", "कृपया",
            "जानकारी", "दे", "दो", "कितना", "मिलेगा", "दर", "दाम", "आज", "मुझे", "मैं", "हम", "मेरा",
            "मेरी", "मेरे", "पास", "कौन", "कौनसी", "सी", "सा", "हैं", "बेचूँ", "बेचनी", "चाहता", "चाहती", "हूँ", "हूं",
        }
        for sw in stop_words:
            clean = re.sub(r"(?<!\w)" + re.escape(sw) + r"(?!\w)", " ", clean, flags=re.IGNORECASE)

        # Clean punctuation and extra whitespace
        clean = re.sub(r"[?!,.:;\"'(){}\[\]]", " ", clean)
        tokens = [t.strip() for t in clean.split() if t.strip().lower() not in stop_words and len(t.strip()) > 1]
        clean_loc = " ".join(tokens).strip()

        # If meaningful location string remains (at least 2 characters)
        if len(clean_loc) >= 2 and not clean_loc.isnumeric():
            return clean_loc

        return None

    def parse_query(self, text: str) -> Dict[str, Any]:
        """Perform dynamic vernacular NLU parsing to extract intent and entities.

        Args:
            text: Farmer's query string in Hindi or English.

        Returns:
            Dict containing parsed intent, commodity, quantity_quintals, location, and raw query.
        """
        commodity = self._extract_commodity(text)
        quantity = self._extract_quantity(text)
        location = self._extract_location(text, commodity=commodity, quantity=quantity)

        clean_lower = text.lower()
        has_devanagari = any("\u0900" <= c <= "\u097f" for c in text)
        english_words = {"what", "is", "the", "best", "where", "can", "i", "sell", "price", "profit", "for", "in", "of", "and", "how", "much"}
        words_set = set(re.findall(r"\b[a-zA-Z]+\b", clean_lower))
        has_english_tokens = len(words_set.intersection(english_words)) >= 2
        is_hindi = has_devanagari or (not has_english_tokens and any(w in clean_lower for w in ["namaste", "bhav", "gehun", "aalu", "kisan", "pyaj", "chawal", "batao"]))

        # Classify intent
        if any(w in clean_lower for w in ["hello", "hi", "namaste", "नमस्ते", "प्रणाम", "राम राम", "सत श्री अकाल"]):
            if not commodity and not location:
                intent = "GREETING"
            else:
                intent = "ARBITRAGE_RECOMMENDATION"
        elif any(w in clean_lower for w in ["weather", "rain", "मौसम", "बारिश", "तापमान", "clouds"]):
            intent = "WEATHER"
        elif any(w in clean_lower for w in ["help", "kaise", "how to", "मदद", "सहायता"]):
            intent = "HELP"
        elif commodity and location:
            intent = "ARBITRAGE_RECOMMENDATION"
        elif commodity:
            intent = "MANDI_PRICE"
        else:
            intent = "GENERAL_QUERY"

        return {
            "raw_text": text,
            "intent": intent,
            "commodity": commodity,
            "quantity_quintals": quantity if quantity else 50.0,
            "location": location,
            "is_hindi": is_hindi,
        }

    def synthesize_speech(
        self,
        text: str,
        language: str = "hi",
    ) -> Dict[str, Any]:
        """Synthesize spoken audio from text using gTTS.

        Args:
            text: Text script to convert into voice audio.
            language: Language code ('hi' for Hindi, 'en' for English).

        Returns:
            Dict containing status, audio_bytes, base64 data URI, and mime type.
        """
        if not HAS_GTTS:
            return {
                "status": "warning",
                "message": "gTTS not installed. Audio synthesis skipped.",
                "audio_bytes": b"",
                "audio_url": "",
                "mime_type": "audio/mp3",
            }

        # Strip special markdown symbols so speech flows naturally
        speech_text = re.sub(r"[*_~`#\[\]]", "", text)
        speech_text = re.sub(r"₹\s*", "रुपये ", speech_text) if language == "hi" else re.sub(r"₹\s*", "Rupees ", speech_text)
        speech_text = re.sub(r"\s+", " ", speech_text).strip()

        lang_code = "hi" if "hi" in language.lower() else "en"

        try:
            tts = gTTS(text=speech_text, lang=lang_code, slow=False)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            audio_bytes = buffer.getvalue()
            b64_uri = "data:audio/mp3;base64," + base64.b64encode(audio_bytes).decode("utf-8")
            return {
                "status": "success",
                "audio_bytes": audio_bytes,
                "audio_url": b64_uri,
                "mime_type": "audio/mp3",
                "language": lang_code,
            }
        except Exception as exc:
            return {
                "status": "warning",
                "message": f"gTTS audio synthesis unavailable: {exc}",
                "audio_bytes": b"",
                "audio_url": "",
                "mime_type": "audio/mp3",
                "language": lang_code,
            }

    def execute_arbitrage_pipeline(
        self,
        location: str,
        commodity: str,
        quantity_quintals: float,
    ) -> Dict[str, Any]:
        """Execute the live Scout -> Predictor -> Planner multi-agent pipeline."""
        loc_info = self.planner.geocode_farmer_location(location)
        farmer_lat = loc_info["latitude"]
        farmer_lon = loc_info["longitude"]
        state = loc_info["state"]
        district = loc_info.get("district", location)

        # Scout live weather & agro-climatic NDVI
        weather = self.scout.fetch_weather(farmer_lat, farmer_lon)
        ndvi = self.predictor.generate_ndvi_index(state, commodity, district=district)

        # Discover candidate APMC mandis
        candidates = self.planner.discover_candidate_markets(
            farmer_lat=farmer_lat,
            farmer_lon=farmer_lon,
            state=state,
            commodity=commodity,
        )

        if not candidates:
            raise RuntimeError(f"No candidate APMC markets found for crop '{commodity}' in '{state}'.")

        # Predict ML quantile prices
        predictions = {}
        for cm in candidates:
            raw_mname = cm.get("market_name") or "APMC Market"
            m_name = str(raw_mname).strip()
            m_key = m_name.lower()

            m_modal = cm.get("modal_price")
            try:
                modal_val = float(m_modal) if m_modal is not None and not (isinstance(m_modal, float) and math.isnan(m_modal)) else 2400.0
            except (ValueError, TypeError):
                modal_val = 2400.0

            mandi_rec = {
                "state": state,
                "district": str(cm.get("district") or district),
                "market": m_name,
                "commodity": commodity,
                "variety": str(cm.get("variety") or "Local"),
                "modal_price": modal_val,
                "arrival_date": cm.get("arrival_date"),
            }
            preds = self.predictor.predict_quantile_prices(
                mandi_record=mandi_rec,
                weather_data=weather,
                ndvi_data=ndvi,
            )
            predictions[m_key] = preds

        # Highway logistics and profit optimization
        logistics = self.planner.optimize_logistics(
            farmer_lat=farmer_lat,
            farmer_lon=farmer_lon,
            predictions_by_market=predictions,
            quantity_quintals=quantity_quintals,
            state=state,
            target_markets=candidates,
        )

        if not logistics:
            raise RuntimeError("Logistics calculation returned no feasible routes.")

        logistics.sort(key=lambda x: x.get("net_expected_profit_inr", 0), reverse=True)
        winner = logistics[0]

        return {
            "winner": winner,
            "all_markets": logistics,
            "farmer_lat": farmer_lat,
            "farmer_lon": farmer_lon,
            "state": state,
            "district": district,
            "weather": weather,
        }

    def process_voice_query(
        self,
        query_input: Union[str, bytes, Path],
        input_type: str = "text",
        language: str = "hi-IN",
    ) -> Dict[str, Any]:
        """Main end-to-end voice processing and recommendation pipeline.

        Args:
            query_input: User spoken audio (bytes/file) or raw text string.
            input_type: 'text' or 'audio'.
            language: Preferred language ('hi-IN' or 'en-IN').

        Returns:
            Comprehensive result dictionary with transcript, entities, response script, and audio.
        """
        # Step 1: Speech-to-Text if audio input
        if input_type == "audio":
            stt_res = self.transcribe_audio(query_input, language=language)
            if stt_res["status"] == "success" and stt_res.get("transcript"):
                query_text = stt_res["transcript"]
            else:
                query_text = (
                    "मुजफ्फरनगर उत्तर प्रदेश में 50 क्विंटल गेहूं के लिए मंडी भाव"
                    if "hi" in language
                    else "Best mandi price for 50 quintals Wheat in Muzaffarnagar"
                )
        else:
            query_text = str(query_input).strip()

        # Step 2: Vernacular NLU parsing
        parsed = self.parse_query(query_text)
        commodity = parsed.get("commodity") or "Wheat"
        quantity = parsed.get("quantity_quintals") or 50.0
        location = parsed.get("location") or "Muzaffarnagar, Uttar Pradesh"
        is_hindi = parsed.get("is_hindi", True) or "hi" in language

        # Step 3: Multi-agent execution
        try:
            pipeline_data = self.execute_arbitrage_pipeline(
                location=location,
                commodity=commodity,
                quantity_quintals=quantity,
            )
            winner = pipeline_data.get("winner")
        except Exception as exc:
            winner = None
            pipeline_data = {"error": str(exc)}

        # Step 4: Construct spoken script and markdown reply
        if winner:
            m_name = winner.get("market_name", "Nearest APMC")
            dist_km = winner.get("one_way_distance_km", 0.0)
            modal_p = (
                winner.get("p50_median_price")
                or winner.get("p50_expected")
                or winner.get("predicted_modal_price")
                or winner.get("expected_modal_price_inr")
                or winner.get("modal_price", 0.0)
            )
            net_profit = winner.get("net_expected_profit_inr", 0.0)
            trans_cost = winner.get("total_transport_cost_inr", 0.0)
            truck = winner.get("truck_type") or winner.get("vehicle_assigned", "Freight Vehicle")

            if is_hindi:
                spoken_text = (
                    f"नमस्ते किसान भाई! {location} क्षेत्र में {quantity:g} क्विंटल {commodity} के लिए सबसे लाभकारी मंडी "
                    f"{m_name} पाई गई है, जो आपसे {dist_km:.1f} किलोमीटर दूर है। "
                    f"यहाँ अपेक्षित मंडी भाव ₹{modal_p:,.0f} प्रति क्विंटल है। "
                    f"{truck} से कुल परिवहन खर्च ₹{trans_cost:,.0f} घटाने के बाद आपका शुद्ध लाभ ₹{net_profit:,.0f} होगा।"
                )
                markdown_text = (
                    f"🌾 **नमस्ते किसान भाई!**\n\n"
                    f"📍 **उत्तम मंडी**: **{m_name}** ({dist_km:.1f} km)\n"
                    f"💰 **अपेक्षित भाव**: **₹{modal_p:,.0f}** / क्विंटल\n"
                    f"🚛 **अनुशंसित वाहन**: {truck} (परिवहन खर्च: ₹{trans_cost:,.0f})\n"
                    f"📈 **कुल शुद्ध जेब लाभ**: **₹{net_profit:,.0f}**\n\n"
                    f"क्या आप इस रिपोर्ट को WhatsApp या SMS पर प्राप्त करना चाहते हैं?"
                )
            else:
                spoken_text = (
                    f"Hello farmer! For your {quantity:g} quintals of {commodity} near {location}, the optimal market is "
                    f"{m_name}, located {dist_km:.1f} kilometers away. "
                    f"Expected modal price is ₹{modal_p:,.0f} per quintal. "
                    f"After logistics expenses of ₹{trans_cost:,.0f} using a {truck}, your net pocket profit is ₹{net_profit:,.0f}."
                )
                markdown_text = (
                    f"🌾 **Hello Farmer!**\n\n"
                    f"📍 **Optimal Mandi**: **{m_name}** ({dist_km:.1f} km)\n"
                    f"💰 **Expected Price**: **₹{modal_p:,.0f}** / quintal\n"
                    f"🚛 **Assigned Fleet**: {truck} (Transport Cost: ₹{trans_cost:,.0f})\n"
                    f"📈 **Net Pocket Profit**: **₹{net_profit:,.0f}**\n\n"
                    f"Would you like to dispatch this report via WhatsApp or SMS?"
                )
        else:
            err_msg = pipeline_data.get("error", "Location or crop information unavailable.")
            if is_hindi:
                spoken_text = (
                    f"नमस्ते किसान भाई! {location} में {commodity} की जानकारी खोजने में सहायता की जा रही है। "
                    f"कृपया जिले का नाम या फसल पुनः बताएं।"
                )
                markdown_text = (
                    f"🌾 **नमस्ते किसान भाई!**\n\n"
                    f"हम {location} में {commodity} के लिए मंडी डेटा प्रोसेस कर रहे हैं। ({err_msg})\n"
                    f"कृपया अपना सही जिला एवं फसल स्पष्ट करें (जैसे: *'मुजफ्फरनगर 50 क्विंटल गेहूं'*)."
                )
            else:
                spoken_text = (
                    f"Hello farmer! Mandi intelligence lookup active for {commodity} near {location}. "
                    f"Please confirm your exact district name and crop quantity."
                )
                markdown_text = (
                    f"🌾 **Hello Farmer!**\n\n"
                    f"Processing spatial arbitrage for {commodity} near {location}. ({err_msg})\n"
                    f"Please verify your district name (e.g. *'Muzaffarnagar 50 quintals Wheat'*)."
                )

        # Step 5: Synthesize spoken audio response
        tts_res = self.synthesize_speech(spoken_text, language="hi" if is_hindi else "en")

        return {
            "status": "success",
            "transcript": query_text,
            "intent": parsed.get("intent", "ARBITRAGE_RECOMMENDATION"),
            "entities": {
                "commodity": commodity,
                "quantity_quintals": quantity,
                "location": location,
            },
            "response_text": spoken_text,
            "reply_markdown": markdown_text,
            "audio_bytes": tts_res.get("audio_bytes", b""),
            "audio_url": tts_res.get("audio_url", ""),
            "mime_type": tts_res.get("mime_type", "audio/mp3"),
            "pipeline_data": pipeline_data,
        }

    def chat(
        self,
        message: str,
        session_state: Optional[Dict[str, Any]] = None,
        language: str = "hi-IN",
    ) -> Dict[str, Any]:
        """Multi-turn conversational chatbot interface for webpage farmers.

        Args:
            message: Incoming farmer message from chat UI.
            session_state: Dictionary holding multi-turn memory (crop, quantity, location).
            language: Preferred language code ('hi-IN' or 'en-IN').

        Returns:
            Dict containing reply_text, reply_markdown, audio_url, session_state, and recommendation.
        """
        state = dict(session_state or {})
        parsed = self.parse_query(message)
        is_hindi = parsed.get("is_hindi", True) or "hi" in language

        # Merge extracted entities into state memory
        if parsed.get("commodity"):
            state["commodity"] = parsed["commodity"]
        if parsed.get("quantity_quintals"):
            state["quantity_quintals"] = parsed["quantity_quintals"]
        if parsed.get("location"):
            state["location"] = parsed["location"]

        # Handle Greetings / Small talk
        if parsed.get("intent") == "GREETING" and not state.get("commodity") and not state.get("location"):
            if is_hindi:
                reply = (
                    "राम राम किसान भाई! 🙏 मैं आपका **K.I.S.A.N. AI मित्र** हूँ। "
                    "मैं आपको आपकी फसल के लिए सबसे अधिक मुनाफा देने वाली मंडी, सही किराया और मौसम की सटीक जानकारी देता हूँ।\n\n"
                    "👉 आप कौन सी फसल बेचना चाहते हैं और आपके पास कितनी पैदावार (क्विंटल) है?"
                )
                spoken = "राम राम किसान भाई! मैं आपका किसान एआई मित्र हूँ। आप कौन सी फसल बेचना चाहते हैं और आपकी पैदावार कितनी है?"
            else:
                reply = (
                    "Hello Farmer! 🙏 I am your **K.I.S.A.N. AI Mitra** assistant. "
                    "I find the highest-profit APMC mandi for your harvest, calculate real diesel transport costs, and maximize your net income.\n\n"
                    "👉 What crop are you selling, and what is your yield quantity in quintals?"
                )
                spoken = "Hello farmer! I am your Kisan AI assistant. What crop are you planning to sell and what is your yield quantity?"

            tts = self.synthesize_speech(spoken, language="hi" if is_hindi else "en")
            return {
                "reply_text": reply,
                "spoken_text": spoken,
                "audio_url": tts.get("audio_url", ""),
                "session_state": state,
                "recommendation": None,
                "entities": state,
            }

        # Handle Weather query
        if parsed.get("intent") == "WEATHER" and state.get("location"):
            loc_name = state["location"]
            try:
                loc_info = self.planner.geocode_farmer_location(loc_name)
                w = self.scout.fetch_weather(loc_info["latitude"], loc_info["longitude"])
                temp = w.get("temperature", 28.0)
                cond = w.get("weather_condition", "Clear")
                rain = w.get("precipitation", 0.0)
                if is_hindi:
                    reply = (
                        f"🌦️ **{loc_name} का मौसम पूर्वानुमान**:\n"
                        f"• तापमान: **{temp}°C**\n"
                        f"• मौसम: **{cond}**\n"
                        f"• बारिश की संभावना: **{rain} mm**\n\n"
                        f"अपनी फसल की बिक्री के लिए सबसे अच्छी मंडी जानने के लिए फसल का नाम बताएं।"
                    )
                    spoken = f"{loc_name} में वर्तमान तापमान {temp} डिग्री सेल्सियस है और मौसम {cond} है।"
                else:
                    reply = (
                        f"🌦️ **Weather Forecast for {loc_name}**:\n"
                        f"• Temperature: **{temp}°C**\n"
                        f"• Condition: **{cond}**\n"
                        f"• Rainfall: **{rain} mm**\n\n"
                        f"To find the highest-paying mandi, let me know your crop name!"
                    )
                    spoken = f"Current temperature in {loc_name} is {temp} degrees Celsius and condition is {cond}."

                tts = self.synthesize_speech(spoken, language="hi" if is_hindi else "en")
                return {
                    "reply_text": reply,
                    "spoken_text": spoken,
                    "audio_url": tts.get("audio_url", ""),
                    "session_state": state,
                    "recommendation": None,
                    "entities": state,
                }
            except Exception:
                pass

        # Check missing entities for Spatial Arbitrage Optimization
        missing = []
        if not state.get("commodity"):
            missing.append("फसल का नाम (जैसे गेहूं, आलू, प्याज)" if is_hindi else "crop name (e.g. Wheat, Potato, Onion)")
        if not state.get("location"):
            missing.append("आपका जिला या शहर" if is_hindi else "your district or town")
        if not state.get("quantity_quintals"):
            missing.append("पैदावार (क्विंटल में)" if is_hindi else "quantity in quintals")

        # If any essential entity is missing, prompt farmer gently
        if missing:
            if is_hindi:
                reply = (
                    f"कृपया अपनी **{', '.join(missing)}** बताएं ताकि मैं सभी निकटवर्ती मंडियों के भाव और वाहन खर्च का हिसाब लगाकर सर्वोत्तम मंडी बता सकूँ।"
                )
                spoken = f"कृपया अपनी {', '.join(missing)} बताएं ताकि मैं सबसे अच्छी मंडी बता सकूँ।"
            else:
                reply = (
                    f"Please specify your **{', '.join(missing)}** so I can calculate roadway fuel freight and find your highest profit APMC mandi!"
                )
                spoken = f"Please specify your {', '.join(missing)} so I can find the best market for you."

            tts = self.synthesize_speech(spoken, language="hi" if is_hindi else "en")
            return {
                "reply_text": reply,
                "spoken_text": spoken,
                "audio_url": tts.get("audio_url", ""),
                "session_state": state,
                "recommendation": None,
                "entities": state,
            }

        # All entities available! Execute Full Multi-Agent Arbitrage Pipeline
        crop = state["commodity"]
        qty = float(state.get("quantity_quintals", 50.0))
        loc = state["location"]

        try:
            pipeline_data = self.execute_arbitrage_pipeline(
                location=loc,
                commodity=crop,
                quantity_quintals=qty,
            )
            winner = pipeline_data.get("winner")
        except Exception as exc:
            winner = None
            pipeline_data = {"error": str(exc)}

        if winner:
            m_name = winner.get("market_name", "APMC Mandi")
            dist_km = winner.get("one_way_distance_km", 0.0)
            modal_p = (
                winner.get("p50_median_price")
                or winner.get("p50_expected")
                or winner.get("predicted_modal_price")
                or winner.get("expected_modal_price_inr")
                or winner.get("modal_price", 0.0)
            )
            net_profit = winner.get("net_expected_profit_inr", 0.0)
            trans_cost = winner.get("total_transport_cost_inr", 0.0)
            vehicle = winner.get("truck_type") or winner.get("vehicle_assigned", "Freight Vehicle")

            if is_hindi:
                spoken = (
                    f"किसान भाई! {loc} में {qty:g} क्विंटल {crop} के लिए सबसे लाभकारी मंडी {m_name} है, "
                    f"जो {dist_km:.1f} किलोमीटर दूर है। यहाँ अपेक्षित भाव ₹{modal_p:,.0f} प्रति क्विंटल है। "
                    f"परिवहन खर्च घटाकर आपका कुल शुद्ध लाभ ₹{net_profit:,.0f} होगा।"
                )
                reply = (
                    f"🎉 **सर्वोत्तम मंडी मिल गई!**\n\n"
                    f"📍 **अनुशंसित मंडी**: **{m_name}**\n"
                    f"🛣️ **दूरी**: **{dist_km:.1f} किमी** (एकतरफा)\n"
                    f"💰 **अपेक्षित भाव**: **₹{modal_p:,.0f}** / क्विंटल\n"
                    f"🚛 **अनुशंसित वाहन**: {vehicle}\n"
                    f"⛽ **परिवहन खर्च**: ₹{trans_cost:,.0f}\n"
                    f"💵 **अनुमानित शुद्ध लाभ (Net Profit)**: **₹{net_profit:,.0f}**\n\n"
                    f"👉 आप इसे डैशबोर्ड पर भी देख सकते हैं या मुझे कोई अन्य फसल पूछ सकते हैं!"
                )
            else:
                spoken = (
                    f"Farmer! For your {qty:g} quintals of {crop} near {loc}, the best APMC market is {m_name}, "
                    f"{dist_km:.1f} km away with an expected price of ₹{modal_p:,.0f} per quintal. "
                    f"After freight expenses, your net expected profit will be ₹{net_profit:,.0f}."
                )
                reply = (
                    f"🎉 **Optimal APMC Mandi Resolved!**\n\n"
                    f"📍 **Recommended Mandi**: **{m_name}**\n"
                    f"🛣️ **Distance**: **{dist_km:.1f} km**\n"
                    f"💰 **Expected Modal Price**: **₹{modal_p:,.0f}** / quintal\n"
                    f"🚛 **Vehicle Fleet**: {vehicle}\n"
                    f"⛽ **Freight Overhead**: ₹{trans_cost:,.0f}\n"
                    f"💵 **Net Expected Pocket Profit**: **₹{net_profit:,.0f}**\n\n"
                    f"👉 You can view full routing breakdown on the dashboard or ask about another crop!"
                )

            rec_payload = {
                "market_name": m_name,
                "one_way_distance_km": dist_km,
                "expected_modal_price_inr": modal_p,
                "net_expected_profit_inr": net_profit,
                "total_transport_cost_inr": trans_cost,
                "truck_type": vehicle,
                "crop": crop,
                "quantity": qty,
                "location": loc,
            }
        else:
            err = pipeline_data.get("error", "No markets found")
            if is_hindi:
                spoken = f"क्षमा करें किसान भाई, {loc} में {crop} के लिए मंडी विवरण खोजने में त्रुटि हुई।"
                reply = f"⚠️ **सूचना**: {loc} के निकट {crop} का डेटा प्राप्त नहीं हो सका ({err})। कृपया नजदीकी मुख्य जिला या राज्य का नाम पुनः बताएं।"
            else:
                spoken = f"Sorry farmer, could not resolve mandi details for {crop} near {loc}."
                reply = f"⚠️ **Notice**: Could not find APMC data for {crop} near {loc} ({err}). Please verify the district name."
            rec_payload = None

        tts = self.synthesize_speech(spoken, language="hi" if is_hindi else "en")

        return {
            "reply_text": reply,
            "spoken_text": spoken,
            "audio_url": tts.get("audio_url", ""),
            "session_state": state,
            "recommendation": rec_payload,
            "entities": state,
        }
