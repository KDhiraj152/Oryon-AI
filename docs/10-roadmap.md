# Roadmap

---

**Author:** K Dhiraj
**Email:** k.dhiraj.srihari@gmail.com
**Version:** 4.1.0
**Last Updated:** February 11, 2026

---

## Strategic Roadmap

The development roadmap focuses on three core objectives:

1. **Scale** – Serve 10 million concurrent users
2. **Personalize** – Adaptive learning at individual user level
3. **Localize** – Deep integration with multilingual AIal ecosystem

---

## Q1 2026: Foundation Scaling

### Kubernetes Auto-scaling

**Objective:** Handle variable load from low-traffic to peak periods.

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: shiksha-setu-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: shiksha-setu-backend
  minReplicas: 3
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
```

**Expected Outcome:**
- Auto-scale during peak hours (8 AM - 4 PM IST)
- Reduce infrastructure cost by 60% during off-hours
- Handle 100x traffic spikes during exam season

### Redis Cluster Migration

**Objective:** Transition from single Redis to clustered deployment.

```python
# backend/cache/redis_cluster.py
from redis.cluster import RedisCluster

class ClusteredCacheService:
    def __init__(self):
        self.cluster = RedisCluster(
            startup_nodes=[
                {"host": "redis-node-1", "port": 6379},
                {"host": "redis-node-2", "port": 6379},
                {"host": "redis-node-3", "port": 6379},
            ],
            decode_responses=True,
            read_from_replicas=True,
        )

    async def get_with_fallback(self, key: str, fallback_fn):
        """Get from cache with automatic fallback to read replicas."""
        try:
            value = await self.cluster.get(key)
            if value:
                return json.loads(value)
        except RedisClusterException:
            # Fallback to any available replica
            value = await self.cluster.get(key, target_nodes="replicas")

        if not value:
            value = await fallback_fn()
            await self.cluster.set(key, json.dumps(value), ex=3600)

        return value
```

**Expected Outcome:**
- 6-node cluster with 3 masters, 3 replicas
- 99.99% cache availability
- Sub-millisecond read latency via replica reads

### Database Read Replicas

**Objective:** Scale read operations for analytics and reporting.

```python
# backend/database.py
class DatabaseManager:
    def __init__(self):
        self.write_engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=20,
            max_overflow=10,
        )
        self.read_engine = create_async_engine(
            settings.DATABASE_READ_URL,  # Read replica
            pool_size=50,
            max_overflow=30,
        )

    def session(self, readonly: bool = False):
        engine = self.read_engine if readonly else self.write_engine
        return AsyncSession(engine)
```

---

## Q2 2026: Intelligence Enhancement

### Multi-modal Understanding

**Objective:** Process images, diagrams, and handwritten content.

```python
# backend/services/multimodal.py
from transformers import AutoProcessor, AutoModelForVision2Seq

class MultimodalService:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/kosmos-2-patch14-224"
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            "microsoft/kosmos-2-patch14-224",
            device_map="auto",
            torch_dtype=torch.float16,
        )

    async def analyze_diagram(
        self,
        image: bytes,
        question: str,
    ) -> DiagramAnalysis:
        """Analyze diagrams with question context."""
        inputs = self.processor(
            text=f"<grounding> {question}",
            images=Image.open(io.BytesIO(image)),
            return_tensors="pt",
        )

        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(generated_ids[0])

        return DiagramAnalysis(
            description=text,
            detected_elements=self._extract_entities(text),
            content_context=await self._map_to_content_domain(text),
        )
```

**Use Cases:**
- Analyze geometry diagrams for math homework
- Extract text from handwritten notes
- Explain science diagrams (cell structure, circuits)

### Adaptive Learning Engine

**Objective:** Personalize content difficulty based on user performance.

```python
# backend/services/adaptive.py
class AdaptiveLearningEngine:
    def __init__(self):
        self.knowledge_graph = KnowledgeGraphService()
        self.assessment_service = AssessmentService()

    async def recommend_next(
        self,
        user_ref_id: str,
        subject: str,
    ) -> LearningRecommendation:
        """Generate personalized learning path."""
        # Get student mastery levels
        mastery = await self.assessment_service.get_mastery_levels(
            user_ref_id, subject
        )

        # Find prerequisite gaps
        gaps = await self.knowledge_graph.find_gaps(mastery)

        # Select optimal next concept
        next_concept = await self._select_next_concept(
            mastery=mastery,
            gaps=gaps,
            user_preference=await self._detect_style(user_ref_id),
        )

        return LearningRecommendation(
            concept=next_concept,
            difficulty=self._calculate_difficulty(mastery, next_concept),
            content_type=self._select_content_type(user_ref_id),
            estimated_time=next_concept.average_time,
        )

    def _calculate_difficulty(
        self,
        mastery: dict,
        concept: Concept,
    ) -> float:
        """Zone of proximal development targeting."""
        prereq_mastery = np.mean([
            mastery.get(p, 0.5) for p in concept.prerequisites
        ])

        # Target difficulty: 15-20% above current mastery
        return min(prereq_mastery + 0.17, 1.0)
```

### Knowledge Graph

**Objective:** Map content_domain concepts with prerequisite relationships.

```python
# backend/services/knowledge_graph.py
class KnowledgeGraphService:
    def __init__(self):
        self.graph = Neo4jGraph(settings.NEO4J_URL)

    async def find_learning_path(
        self,
        start_concept: str,
        target_concept: str,
    ) -> list[Concept]:
        """Find optimal path through knowledge graph."""
        query = """
        MATCH path = shortestPath(
            (start:Concept {id: $start})-[:PREREQUISITE*]->(end:Concept {id: $target})
        )
        RETURN nodes(path) as concepts
        """

        result = await self.graph.run(query, start=start_concept, target=target_concept)
        return [Concept(**node) for node in result["concepts"]]

    async def get_concept_cluster(
        self,
        concept_id: str,
        depth: int = 2,
    ) -> ConceptCluster:
        """Get related concepts for contextual learning."""
        query = """
        MATCH (c:Concept {id: $id})
        OPTIONAL MATCH (c)-[:RELATED_TO*1..$depth]-(related:Concept)
        OPTIONAL MATCH (c)<-[:PREREQUISITE]-(prereq:Concept)
        OPTIONAL MATCH (c)-[:PREREQUISITE]->(builds_to:Concept)
        RETURN c, collect(DISTINCT related) as related,
               collect(DISTINCT prereq) as prerequisites,
               collect(DISTINCT builds_to) as builds_to
        """

        return await self.graph.run(query, id=concept_id, depth=depth)
```

---

## Q3 2026: Ecosystem Integration

### Multi-Provider Content Integration

**Objective:** Support content from multiple domain-specific providers and knowledge bases.

**Target Providers:**
1. Custom Knowledge Base (primary)
2. Public API Integrations
3. Regional Content Providers
4. Enterprise Content Sources
5. Community-Contributed Domains

```python
# backend/services/content_domain.py
class ContentMappingService:
    PROVIDER_CONFIGS = {
        "primary": {
            "content_url": "https://api.example.com/content",
            "content_pattern": "modular",
            "scoring_system": "weighted",
        },
        "enterprise": {
            "content_url": "https://enterprise.example.com/api",
            "content_pattern": "hierarchical",
            "scoring_system": "percentage",
        },
        # Additional providers...
    }

    async def map_content(
        self,
        content_id: str,
        target_providers: list[str],
    ) -> dict[str, ContentMapping]:
        """Map content across multiple providers."""
        mappings = {}

        for provider in target_providers:
            config = self.PROVIDER_CONFIGS[provider]
            catalog = await self._load_catalog(provider)

            mapping = await self._find_mapping(
                content_id=content_id,
                catalog=catalog,
                content_pattern=config["content_pattern"],
            )
            mappings[provider] = mapping

        return mappings
```

### Offline Mode

**Objective:** Enable learning in low-connectivity regions.

```typescript
// frontend/src/workers/offlineSync.ts
class OfflineSyncManager {
  private db: IDBDatabase;
  private syncQueue: SyncOperation[] = [];

  async init(): Promise<void> {
    this.db = await this.openDatabase();
    await this.registerServiceWorker();

    window.addEventListener('online', () => this.syncToServer());
  }

  async cacheContent(contentId: string): Promise<void> {
    // Download content for offline use
    const content = await api.getContent(contentId);

    await this.storeLocally({
      type: 'content',
      id: contentId,
      data: content,
      cachedAt: Date.now(),
      expiresAt: Date.now() + 7 * 24 * 60 * 60 * 1000, // 7 days
    });

    // Also cache related assets
    for (const asset of content.assets) {
      await this.cacheAsset(asset);
    }
  }

  async submitOfflineAnswer(
    questionId: string,
    answer: string,
  ): Promise<void> {
    const operation: SyncOperation = {
      type: 'answer',
      questionId,
      answer,
      timestamp: Date.now(),
      synced: false,
    };

    await this.storeLocally(operation);
    this.syncQueue.push(operation);

    if (navigator.onLine) {
      await this.syncToServer();
    }
  }

  private async syncToServer(): Promise<void> {
    const pending = this.syncQueue.filter(op => !op.synced);

    for (const operation of pending) {
      try {
        await this.executeSync(operation);
        operation.synced = true;
        await this.updateLocal(operation);
      } catch (error) {
        console.error('Sync failed, will retry:', error);
      }
    }
  }
}
```

### Admin Dashboard

**Objective:** Provide educators with actionable insights.

```typescript
// frontend/src/components/AdminDashboard.tsx
interface ClassAnalytics {
  class_id: string;
  users: UserProgress[];
  topic_mastery: TopicMastery[];
  attention_alerts: AttentionAlert[];
  recommended_interventions: Intervention[];
}

const AdminDashboard: React.FC = () => {
  const [analytics, setAnalytics] = useState<ClassAnalytics | null>(null);

  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Class Overview */}
      <Card className="col-span-8">
        <CardHeader>
          <CardTitle>Class Progress</CardTitle>
        </CardHeader>
        <CardContent>
          <ProgressHeatmap data={analytics?.topic_mastery} />
        </CardContent>
      </Card>

      {/* Attention Alerts */}
      <Card className="col-span-4">
        <CardHeader>
          <CardTitle>Users Needing Support</CardTitle>
        </CardHeader>
        <CardContent>
          {analytics?.attention_alerts.map(alert => (
            <AttentionAlertCard key={alert.user_ref_id} alert={alert} />
          ))}
        </CardContent>
      </Card>

      {/* AI Recommendations */}
      <Card className="col-span-12">
        <CardHeader>
          <CardTitle>Recommended Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <InterventionList
            interventions={analytics?.recommended_interventions}
            onAssign={handleAssignIntervention}
          />
        </CardContent>
      </Card>
    </div>
  );
};
```

---

## Q4 2026: Advanced Capabilities

### Voice-First Interface

**Objective:** Enable hands-free learning for accessibility.

```python
# backend/services/voice_assistant.py
class VoiceAssistantService:
    def __init__(self):
        self.stt = WhisperService()
        self.tts = EdgeTTSService()
        self.llm = QAService()
        self.context = ConversationContext()

    async def process_voice_query(
        self,
        audio: bytes,
        session_id: str,
        language: str,
    ) -> VoiceResponse:
        """Process voice query and return audio response."""
        # Transcribe audio
        transcript = await self.stt.transcribe(audio, language=language)

        # Maintain conversation context
        context = await self.context.get(session_id)
        context.add_turn(role="user", content=transcript.text)

        # Generate response
        response = await self.llm.generate(
            question=transcript.text,
            context=context.to_messages(),
            language=language,
        )

        context.add_turn(role="assistant", content=response.answer)
        await self.context.save(session_id, context)

        # Synthesize speech
        audio_response = await self.tts.synthesize(
            text=response.answer,
            language=language,
            voice=await self._select_voice(language),
        )

        return VoiceResponse(
            transcript=transcript.text,
            text_response=response.answer,
            audio=audio_response,
            context_summary=context.summary,
        )
```

### Gamification

**Objective:** Increase engagement through game mechanics.

```python
# backend/services/gamification.py
class GamificationService:
    ACHIEVEMENT_DEFINITIONS = {
        "first_question": Achievement(
            id="first_question",
            name="Curious Mind",
            description="Asked your first question",
            points=10,
            badge_url="/badges/curious_mind.svg",
        ),
        "streak_7": Achievement(
            id="streak_7",
            name="Week Warrior",
            description="7-day learning streak",
            points=50,
            badge_url="/badges/week_warrior.svg",
        ),
        "subject_master": Achievement(
            id="subject_master",
            name="Subject Master",
            description="Achieved 90% mastery in a subject",
            points=200,
            badge_url="/badges/subject_master.svg",
        ),
    }

    async def check_achievements(
        self,
        user_ref_id: str,
        event: LearningEvent,
    ) -> list[Achievement]:
        """Check and award achievements based on learning events."""
        profile = await self.get_profile(user_ref_id)
        newly_earned = []

        for achievement_id, definition in self.ACHIEVEMENT_DEFINITIONS.items():
            if achievement_id in profile.earned_achievements:
                continue

            if await self._check_criteria(profile, event, achievement_id):
                await self._award_achievement(user_ref_id, definition)
                newly_earned.append(definition)

        return newly_earned

    async def get_leaderboard(
        self,
        scope: str,  # "team", "org", "region", "state"
        time_period: str,  # "daily", "weekly", "monthly", "all_time"
    ) -> list[LeaderboardEntry]:
        """Get leaderboard for specified scope and time period."""
        # Privacy-conscious: Only show top N, no personal data for others
        return await self.db.get_leaderboard(
            scope=scope,
            period=time_period,
            limit=10,
            anonymize=True,
        )
```

### AI Assistant Personality

**Objective:** Create engaging, personality-driven tutoring experience.

```python
# backend/services/tutor_personality.py
class AssistantPersonalityService:
    PERSONALITIES = {
        "encouraging": {
            "system_prompt": """You are a warm, encouraging tutor who celebrates
            every effort. Use phrases like 'Great thinking!' and 'You're on the
            right track!' Focus on progress, not perfection.""",
            "response_style": "supportive",
        },
        "socratic": {
            "system_prompt": """You are a Socratic tutor who guides through
            questions rather than direct answers. Ask 'What do you think?' and
            'Can you explain why?' Help users discover answers themselves.""",
            "response_style": "questioning",
        },
        "practical": {
            "system_prompt": """You are a practical tutor who connects concepts
            to real-world applications. Use examples from daily life, sports,
            and technology that users can relate to.""",
            "response_style": "applied",
        },
    }

    async def adapt_response(
        self,
        response: str,
        user_ref_id: str,
        personality: str = None,
    ) -> str:
        """Adapt response based on student's preferred personality."""
        if not personality:
            personality = await self._detect_preference(user_ref_id)

        config = self.PERSONALITIES[personality]

        return await self.llm.rewrite(
            text=response,
            style=config["response_style"],
            system=config["system_prompt"],
        )
```

---

## Technical Debt Reduction

### Priority Items

| Item | Effort | Impact | Timeline |
|------|--------|--------|----------|
| Migrate to AsyncIO throughout | Medium | High | Q1 2026 |
| Implement proper dependency injection | Low | Medium | Q1 2026 |
| Add comprehensive OpenTelemetry tracing | Medium | High | Q2 2026 |
| Refactor monolithic services | High | High | Q2-Q3 2026 |
| Implement event sourcing | High | Medium | Q4 2026 |

### Code Refactoring

```python
# Before: Monolithic service
class UnifiedPipelineService:
    def __init__(self):
        self.rag = RAGService()
        self.translate = TranslationService()
        self.tts = TTSService()
        self.stt = STTService()
        # ... 15 more services

    async def process(self, request):
        # 500 lines of procedural code
        pass

# After: Clean architecture with DI
class QuestionAnswerUseCase:
    def __init__(
        self,
        rag_service: RAGServiceProtocol,
        translation_service: TranslationServiceProtocol,
        speech_service: SpeechServiceProtocol,
    ):
        self.rag = rag_service
        self.translation = translation_service
        self.speech = speech_service

    async def execute(self, request: QuestionRequest) -> AnswerResponse:
        # Single responsibility implementation
        context = await self.rag.retrieve(request.question)
        answer = await self.rag.generate(request.question, context)

        if request.translate_to:
            answer = await self.translation.translate(answer, request.translate_to)

        if request.audio_response:
            audio = await self.speech.synthesize(answer)
            return AnswerResponse(text=answer, audio=audio)

        return AnswerResponse(text=answer)
```

---

## Success Metrics

### Q1 2026 Targets

| Metric | Current | Target |
|--------|---------|--------|
| Concurrent users | 10,000 | 100,000 |
| Response latency (p95) | 850ms | 500ms |
| Cache hit rate | 78% | 92% |
| Test coverage | 90% | 95% |

### Q2 2026 Targets

| Metric | Current | Target |
|--------|---------|--------|
| Learning outcome improvement | Baseline | +15% |
| User engagement (DAU/MAU) | TBD | 60% |
| Content personalization accuracy | N/A | 85% |

### Q3-Q4 2026 Targets

| Metric | Target |
|--------|--------|
| State boards integrated | 8 |
| Offline-capable content | 10,000 lessons |
| Reviewer adoption | 50,000 educators |
| Voice query success rate | 95% |

---

*For contribution details, see Section 11: Contribution Summary.*

---

**K Dhiraj**
k.dhiraj.srihari@gmail.com
