import os
import ast
import asyncio
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount, Activity
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import redis
import azure.cognitiveservices.speech as speechsdk
from rtclient import ResponseCreateMessage, RTLowLevelClient, ResponseCreateParams
from botbuilder.schema import ChannelAccount


class MyBot(ActivityHandler):
    def __init__(self):
        super().__init__()
        
        # Configuration settings
        self._configure_services()
        self._load_and_prepare_data()
        self._initialize_clients()
        self._upload_to_azure_search()

    def _configure_services(self):
        """Configure service credentials and thresholds"""
        # Azure OpenAI
        self.openai_config = {
            "api_key": "8929107a6a6b4f37b293a0fa0584ffc3",
            "api_version": "2023-03-15-preview",
            "endpoint": "https://genral-openai.openai.azure.com/",
            "embedding_model": "text-embedding-ada-002"
        }

        # Azure Search
        self.search_config = {
            "service_name": "mainsearch01",
            "index_name": "id",
            "api_key": "Y6dbb3ljV5z33htXQEMR8ICM8nAHxOpNLwEPwKwKB9AzSeBtGPav"
        }

        # Redis
        self.redis_config = {
            "host": "AiKr.redis.cache.windows.net",
            "port": 6380,
            "password": "OD8wyo8NiVxse6DDkEY19481Xr7ZhQAnfAzCaOZKR2U="
        }

        # Thresholds
        self.semantic_threshold = 3.4
        self.vector_threshold = 0.91

    def _load_and_prepare_data(self):
        """Load and prepare Q&A data"""
        try:
            self.qa_data = pd.read_csv("qa_data.csv", encoding="windows-1256")
            print("✅ CSV file loaded successfully!")
            
            # Normalize and clean data
            self.qa_data.rename(columns=lambda x: x.strip().lower(), inplace=True)
            self.qa_data["question"] = self.qa_data["question"].str.strip().str.lower()
            
            if "id" in self.qa_data.columns:
                self.qa_data["id"] = self.qa_data["id"].astype(str)

            # Generate embeddings if needed
            if "embedding" not in self.qa_data.columns or self.qa_data["embedding"].isnull().all():
                self.qa_data["embedding"] = self.qa_data["question"].apply(self.get_embedding)
                self.qa_data.to_csv("embedded_qa_data.csv", index=False)
                print("✅ Embeddings generated and saved.")
            else:
                self.qa_data["embedding"] = self.qa_data["embedding"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )

        except Exception as e:
            print(f"❌ Data loading error: {e}")
            raise
    def clean_text(text):
        return text.strip(" .،!؛؟").lower()

    def _initialize_clients(self):
        """Initialize all service clients"""
        # Azure OpenAI client
        self.openai_client = openai.AzureOpenAI(
            api_key=self.openai_config["api_key"],
            api_version=self.openai_config["api_version"],
            azure_endpoint=self.openai_config["endpoint"]
        )

        # Azure Search client
        self.search_client = SearchClient(
            endpoint=f"https://{self.search_config['service_name']}.search.windows.net/",
            index_name=self.search_config["index_name"],
            credential=AzureKeyCredential(self.search_config["api_key"])
        )

        # Redis client
        self.redis_client = redis.Redis(
            host=self.redis_config["host"],
            port=self.redis_config["port"],
            password=self.redis_config["password"],
            ssl=True,
            decode_responses=True
        )

    def _upload_to_azure_search(self):
        """Upload documents to Azure Cognitive Search"""
        try:
            documents = self.qa_data.to_dict(orient="records")
            upload_result = self.search_client.upload_documents(documents=documents)
            print(f"✅ Uploaded {len(documents)} documents to Azure Search.")
        except Exception as e:
            print(f"❌ Document upload failed: {e}")

    def get_embedding(self, text):
        """Generate text embedding using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.openai_config["embedding_model"],
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return None

    async def on_message_activity(self, turn_context: TurnContext):
        """Handle incoming messages"""
        user_query = turn_context.activity.text
        response = await self.get_response(user_query)
        await turn_context.send_activity(response)

    def detect_critical_issue(self,text):
        trigger_sentences = [
            "تم اكتشاف اختراق أمني كبير.",
            "تمكن قراصنة من الوصول إلى بيانات حساسة.",
            "هناك هجوم إلكتروني على النظام الخاص بنا.",
            "تم تسريب بيانات المستخدمين إلى الإنترنت.",
            "رصدنا محاولة تصيد إلكتروني ضد موظفينا.",
            "تم استغلال ثغرة أمنية في الشبكة.",
            "هناك محاولة وصول غير مصرح بها إلى الملفات السرية."
        ]

        trigger_embeddings = np.array([self.get_embedding(sent) for sent in trigger_sentences])
        text_embedding = np.array(self.get_embedding(text)).reshape(1, -1)
        similarities = cosine_similarity(text_embedding, trigger_embeddings)
        max_similarity = np.max(similarities)
        if max_similarity > 0.9:
            print("This issue should be passed to a human.")
            return True
        return False

    async def on_members_added_activity(
        self, members_added: ChannelAccount, turn_context: TurnContext
    ):
        """Handle new members added"""
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("مرحبًا! كيف يمكنني مساعدتك اليوم؟")

    async def get_response(self, user_query):
        """Main response generation logic"""
        # Check cache first
        if self.clean_text(user_query) in ["إنهاء", "خروج"]:
            print("👋 Goodbye!")
            return "مع السلامة"
        if self.detect_critical_issue(user_query):
            return "هذه المشكلة تحتاج إلى تدخل بشري. سأقوم بالاتصال بخدمة العملاء لدعمك."
        
        cached = self.check_redis_cache(user_query)
        if cached:
            return cached

        # Try semantic search
        semantic_result = await self.semantic_search(user_query)
        if semantic_result:
            return semantic_result

        # Try vector search
        vector_result = await self.vector_search(user_query)
        if vector_result:
            return vector_result

        # Fallback to GPT-4 realtime
        return await self.get_realtime_response(user_query)

    def check_redis_cache(self, query):
        """Check Redis for cached responses"""
        try:
            return self.redis_client.get(query)
        except Exception as e:
            print(f"❌ Redis error: {e}")
            return None

    async def semantic_search(self, query):
        """Perform semantic search"""
        try:
            results = self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name="my-semantic-config-default",
                top=3
            )
            best_match = next(results, None)
            if best_match and best_match["@search.reranker_score"] >= self.semantic_threshold:
                self.redis_client.set(query, best_match["answer"], ex=3600)
                return best_match["answer"]
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
        return None

    async def vector_search(self, query):
        """Perform vector search"""
        try:
            embedding = self.get_embedding(query)
            vector_query = VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=50,
                fields="embedding"
            )
            results = self.search_client.search(
                vector_queries=[vector_query],
                top=3
            )
            best_match = next(results, None)
            if best_match and best_match["@search.score"] >= self.vector_threshold:
                self.redis_client.set(query, best_match["answer"], ex=3600)
                return best_match["answer"]
        except Exception as e:
            print(f"❌ Vector search error: {e}")
        return None

    async def get_realtime_response(self, user_query):
        """GPT-4 realtime fallback"""
        try:
            async with RTLowLevelClient(
                url="https://general-openai02.openai.azure.com/",
                azure_deployment="gpt-4o-realtime-preview",
                key_credential=AzureKeyCredential("9e76306d48fb4e6684e4094d217695ac")
            ) as client:
                instructions = "أنت رجل عربي. لا اريد اي bold points في الاجابة ولا عنواين مرقمة"
                await client.send(ResponseCreateMessage(
                    response=ResponseCreateParams(
                        modalities={"text"},
                        instructions=f"{instructions} {user_query}"
                    )
                ))
                
                response_text = ""
                while True:
                    message = await client.recv()
                    if message.type == "response.done":
                        break
                    if message.type == "response.text.delta":
                        response_text += message.delta
                
                self.redis_client.set(user_query, response_text, ex=3600)
                return response_text
        except Exception as e:
            print(f"❌ Realtime response failed: {e}")
            return "عذرًا، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى."
