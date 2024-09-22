import pyttsx3

def text_to_speech_pyttsx3(text):
    # Initialize the TTS engine
    engine = pyttsx3.init()
    
    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)
    
    # Convert the text to speech
    engine.say(text)
    
    # Wait for the speech to finish
    engine.runAndWait()

# Example usage
text = "vishwa"
text_to_speech_pyttsx3(text)
