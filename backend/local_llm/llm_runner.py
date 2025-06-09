import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib
import pickle
import os

try:
    from gpt4all import GPT4All
except ImportError:
    raise ImportError("Please install gpt4all: pip install gpt4all")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMRunner:
    """
    Local LLM Runner for AutoLawyer using GPT4All
    Handles model loading, inference, caching, and conversation management
    """
    
    def __init__(self, config_path: str = "backend/local_llm/model_config.json"):
        """
        Initialize the LLM Runner with configuration
        
        Args:
            config_path: Path to model configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.is_loaded = False
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_dir = Path(self.config.get("cache_dir", "backend/local_llm/cache"))
        
        # Create cache directory if it doesn't exist
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LLM Runner initialized with model: {self.config.get('model_name', 'Unknown')}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            # Return default configuration
            return {
                "model_name": "mistral-7b-openorca.Q4_0.gguf",
                "model_path": "models/mistral-7b-openorca.Q4_0.gguf",
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "cache_enabled": True,
                "cache_dir": "backend/local_llm/cache"
            }
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise
    
    def load_model(self) -> bool:
        """
        Load the GPT4All model into memory
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self.is_loaded:
            logger.info("Model already loaded")
            return True
        
        try:
            model_path = self.config["model_path"]
            
            # Check if model file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                logger.info("Please download a GPT4All model and place it in the models/ directory")
                return False
            
            logger.info(f"Loading model from: {model_path}")
            start_time = time.time()
            
            # Initialize GPT4All model
            self.model = GPT4All(model_path)
            
            load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def _generate_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate a unique cache key for the prompt and parameters"""
        # Include prompt and relevant parameters for cache key
        cache_data = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens")),
            "temperature": kwargs.get("temperature", self.config.get("temperature")),
            "top_p": kwargs.get("top_p", self.config.get("top_p")),
            "repeat_penalty": kwargs.get("repeat_penalty", self.config.get("repeat_penalty"))
        }
        
        # Create hash of the cache data
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Retrieve cached response if available"""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    logger.info("Using cached response")
                    return cached_data["response"]
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Save response to cache"""
        if not self.cache_enabled:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            cache_data = {
                "response": response,
                "timestamp": time.time()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using the local LLM
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated response from the model
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load model. Cannot generate response.")
        
        # Generate cache key
        cache_key = self._generate_cache_key(prompt, **kwargs)
        
        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        try:
            # Prepare generation parameters
            generation_params = {
                "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 2048)),
                "temp": kwargs.get("temperature", self.config.get("temperature", 0.7)),
                "top_p": kwargs.get("top_p", self.config.get("top_p", 0.9)),
                "repeat_penalty": kwargs.get("repeat_penalty", self.config.get("repeat_penalty", 1.1))
            }
            
            logger.info("Generating response...")
            start_time = time.time()
            
            # Generate response using GPT4All
            response = self.model.generate(prompt, **generation_params)
            
            generation_time = time.time() - start_time
            logger.info(f"Response generated in {generation_time:.2f} seconds")
            
            # Cache the response
            self._save_to_cache(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise RuntimeError(f"LLM generation failed: {str(e)}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Chat interface compatible with OpenAI-style message format
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated response
        """
        # Convert messages to a single prompt
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        return self.generate(full_prompt, **kwargs)
    
    def unload_model(self):
        """Unload the model from memory"""
        if self.model:
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info("Model unloaded from memory")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.config.get("model_name"),
            "model_path": self.config.get("model_path"),
            "is_loaded": self.is_loaded,
            "cache_enabled": self.cache_enabled,
            "config": self.config
        }
    
    def clear_cache(self) -> bool:
        """Clear the LLM response cache"""
        if not self.cache_enabled:
            return False
        
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cache cleared successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

# Singleton instance for global use
_llm_runner_instance = None

def get_llm_runner() -> LLMRunner:
    """Get or create the global LLM runner instance"""
    global _llm_runner_instance
    if _llm_runner_instance is None:
        _llm_runner_instance = LLMRunner()
    return _llm_runner_instance