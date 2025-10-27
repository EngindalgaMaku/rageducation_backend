# Akıllı Kütüphane RAG Sistemi - AI Resim Üretimi Prompt Rehberi

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Önerilen AI Resim Modelleri](#önerilen-ai-resim-modelleri)
- [Sistem Mimarisi Görselleri](#sistem-mimarisi-görselleri)
- [Pipeline ve Veri Akışı Görselleri](#pipeline-ve-veri-akışı-görselleri)
- [Öğretmen Odaklı Arayüz Görselleri](#öğretmen-odaklı-arayüz-görselleri)
- [Akıllı Kütüphane Sistemi Görselleri](#akıllı-kütüphane-sistemi-görselleri)
- [Bilgi Havuzu Yönetimi Görselleri](#bilgi-havuzu-yönetimi-görselleri)
- [UI/UX Mockup Görselleri](#uiux-mockup-görselleri)
- [Eğitici İnfografik Görselleri](#eğitici-infografik-görselleri)
- [Teknik Diyagram Görselleri](#teknik-diyagram-görselleri)
- [Sunum ve Dokümantasyon Görselleri](#sunum-ve-dokümantasyon-görselleri)
- [Prompt Optimizasyon İpuçları](#prompt-optimizasyon-ipuçları)

## 🎯 Genel Bakış

Bu döküman, **Eğitim Alanında RAG Kullanımı ile Akıllı Kütüphane Sistemi** projesi için görsel içerik oluşturmada kullanılabilecek detaylı AI prompt örnekleri içerir. Sistem, öğretmenlerin kendi derslerine özel bilgi havuzları oluşturmasını ve chatbot tarzında öğrenci-kaynak etkileşimi sağlamasını hedeflemektedir. Her prompt, eğitim kurumları ve akıllı kütüphane ortamları için optimize edilmiştir.

## 🤖 Önerilen AI Resim Modelleri

### **Tier 1: Premium Modeller**

1. **DALL-E 3** (OpenAI)

   - En yüksek kalite
   - Metin anlama yeteneği üstün
   - Karmaşık teknik diyagramlar için ideal
   - **Kullanım:** ChatGPT Plus, OpenAI API

2. **Midjourney v6**
   - Sanatsal kalite çok yüksek
   - Profesyonel sunum görselleri
   - Stil tutarlılığı mükemmel
   - **Kullanım:** Discord bot

### **Tier 2: Kaliteli ve Erişilebilir**

3. **Stable Diffusion XL**

   - Ücretsiz ve açık kaynak
   - Yerel kurulum mümkün
   - Özelleştirilebilir
   - **Kullanım:** ComfyUI, Automatic1111

4. **Leonardo.ai**
   - Kullanıcı dostu arayüz
   - Finetuned modeller
   - Batch üretim imkanı
   - **Kullanım:** Web arayüzü

### **Tier 3: Ücretsiz Alternatiif**

5. **Bing Image Creator**

   - DALL-E 3 tabanlı
   - Günlük limit var
   - Kolay erişim
   - **Kullanım:** Microsoft Edge/Bing

6. **Adobe Firefly**
   - Ticari kullanım güvenli
   - Adobe ekosistemi entegrasyonu
   - **Kullanım:** Adobe Creative Cloud

## 🏗️ Sistem Mimarisi Görselleri

### 1. Akıllı Kütüphane Sistem Mimarisi Diyagramı

**Prompt:**

```
Create a professional system architecture diagram showing an Intelligent Library RAG system for educational institutions. The diagram should include:

- Clean, modern isometric 3D style with soft shadows and library-themed elements
- Color scheme: Deep blue (#1565C0), Golden yellow (#FFB300), Green (#388E3C), Light gray (#F5F5F5)
- Main components arranged in layers with library metaphors:
  * Teacher/Librarian Layer: Content curation dashboard, document approval system
  * Student Interface Layer: Chatbot interface, search portal, mobile app
  * API Layer: Authentication, course-specific routing, usage analytics
  * Intelligent Processing: Teacher knowledge pool builder, RAG Core, Auto-categorization
  * RAG Core: Document Processor, Subject-aware Chunker, Domain-specific Embedder, Context-aware Retriever
  * Knowledge Storage: Course-specific Vector DBs, Metadata with curriculum mapping, Approved content repository
  * AI Services: Ollama LLM for Turkish education, Subject-matter embeddings

Include teacher workflow arrows, student query flows, and administrative oversight connections. Style: Academic institution aesthetic with library and classroom icons. Background: subtle educational pattern with book spines and academic elements.
```

### 2. Öğretmen Bilgi Havuzu Oluşturma Pipeline'ı

**Prompt:**

```
Design an educational workflow showing how teachers create personalized knowledge pools:

- Layout: Circular workflow with teacher at the center
- Style: Modern infographic with education-themed icons and warm colors
- Color scheme: Teacher blue (#1976D2), Content orange (#FF8F00), Approval green (#2E7D32)
- Workflow steps:
  1. Teacher Login (profile icon with badge)
  2. Course Selection (course catalog with subject icons: Math, Science, Literature, etc.)
  3. Content Upload (drag-and-drop interface with file types: PDF lectures, PowerPoint slides, Word documents)
  4. Automatic Processing (AI gear processing documents with Turkish language indicators)
  5. Content Review & Approval (teacher reviewing extracted content with approve/reject buttons)
  6. Knowledge Pool Creation (database formation with course-specific categorization)
  7. Student Access Configuration (privacy settings and class permissions)
  8. Chatbot Activation (AI assistant ready for student queries)
  9. Performance Monitoring (analytics showing student engagement and query patterns)

Include Turkish labels, realistic educational content examples (Turkish literature, mathematics problems, science concepts), and teacher persona workflows. Background: Classroom-inspired subtle pattern.
```

## 📊 Pipeline ve Veri Akışı Görselleri

### 3. Ders-Özel Doküman İşleme Pipeline'ı

**Prompt:**

```
Create a detailed flowchart showing course-specific document processing for intelligent library system:

- Style: Educational flowchart with subject-matter branching
- Color coding:
  * Teacher Actions: Purple gradient (#7B1FA2 to #9C27B0)
  * Subject Classification: Blue gradient (#1976D2 to #2196F3)
  * Processing steps: Orange gradient (#F57C00 to #FF9800)
  * Course Integration: Green gradient (#388E3C to #4CAF50)
  * Quality Control: Red gradient (#D32F2F to #F44336)

- Show the enhanced flow:
  Teacher Upload → Subject Auto-Detection (Math/Science/Literature/History branches) → Format Processing (PDF lecture notes/PowerPoint slides/Word handouts) → Turkish Language Processing → Subject-Aware Chunking → Curriculum Alignment Check → Teacher Review & Approval → Course-Specific Embedding → Subject Vector Storage → Cross-Reference Mapping → Student Access Permissions

Include teacher decision points, subject-specific processing variations, approval workflows, and quality control measures. Add realistic Turkish educational examples for each subject. Background: Academic pattern with subject icons.
```

### 4. Öğrenci Soru-Cevap Akış Diyagramı (Akıllı Kütüphane)

**Prompt:**

```
Design a sequence diagram showing intelligent library student interaction flow:

- Style: Educational UML with library and classroom aesthetics
- Participants: Student (mobile/web), Library Chatbot Interface, Course Router, Teacher's Knowledge Pool, Subject-Specific RAG, Curriculum Validator, Turkish Education LLM
- Color scheme: Student green, Library blue, Course purple, Teacher orange, AI silver
- Enhanced interaction flow:
  1. Student logs in with course credentials
  2. Course context identification (which class/subject)
  3. Natural language question in Turkish
  4. Subject classification (Math question? Literature analysis? Science concept?)
  5. Teacher's approved content search
  6. Curriculum-aligned context retrieval
  7. Educational-appropriate response generation
  8. Source citation with teacher materials
  9. Learning analytics tracking
  10. Optional: Suggest related topics or next questions

Include course-specific routing, Turkish language processing, educational content filtering, and teacher oversight indicators. Background: Library atmosphere with bookshelves pattern. Add realistic Turkish student queries for different subjects.
```

## 🎓 Öğretmen Odaklı Arayüz Görselleri

### 5. Öğretmen Dashboard Mockup

**Prompt:**

```
Create a comprehensive teacher dashboard for intelligent library content management:

- Layout: Professional educator workspace with multi-panel design
- Color scheme: Teacher-friendly (Professional blue: #1565C0, Success green: #2E7D32, Warning amber: #FF8F00, Neutral gray: #F5F5F5)
- Main sections:
  * Header: "Öğretmen İçerik Yönetim Paneli" with school branding and teacher profile
  * Left Sidebar: Course management (active classes, subject selection, student groups)
  * Center Panel: Content upload area with drag-and-drop, processing status, and preview
  * Right Panel: Knowledge pool overview, student query analytics, content performance
  * Bottom Bar: Recently uploaded documents, pending approvals, system notifications

Features to highlight:
- Course-specific content organization (Math 9A, Literature 10B, Chemistry 11C)
- Document approval workflow with preview and edit options
- Student question patterns and frequently asked topics
- Content usage analytics (which materials are most queried)
- Turkish educational content examples for different subjects

Style: Clean, professional education management interface similar to Google Classroom or Moodle admin panels. Include realistic Turkish school course structure and content.
```

### 6. İçerik Onay ve Düzenleme Arayüzü

**Prompt:**

```
Design a content approval and editing interface for teachers:

- Layout: Split-screen with content preview and editing controls
- Left side: Document preview with highlighted extracted sections
- Right side: Approval controls, tagging options, and metadata editing
- Features:
  * Auto-extracted content sections with approve/reject buttons
  * Subject tagging (Matematik, Türkçe, Fen Bilgisi, Sosyal Bilgiler)
  * Difficulty level selection (6. sınıf, 7. sınıf, etc.)
  * Curriculum alignment indicators
  * Custom teacher notes and learning objectives
  * Student accessibility settings

Include Turkish educational content examples, subject-specific icons, and teacher workflow elements. Color coding for approval status (green approved, yellow pending, red needs revision). Style: Educational software interface with clear visual hierarchy.
```

## 📚 Akıllı Kütüphane Sistemi Görselleri

### 7. Kütüphane Etkileşim Haritası

**Prompt:**

```
Create a comprehensive interaction map showing the intelligent library ecosystem:

- Style: Modern infographic with library and school environment
- Central focus: Digital library building icon with smart technology elements
- User personas with distinct pathways:
  * Students: Various age groups (ilkokul, ortaokul, lise) with different device usage
  * Teachers: Subject specialists creating and managing content
  * Librarians: System administrators and content curators
  * Parents: Monitoring student progress and accessing resources

Interaction flows:
- Student journey: Login → Course selection → Question asking → Answer receiving → Further exploration
- Teacher journey: Content creation → Approval process → Student monitoring → Performance analysis
- Librarian journey: System management → Content oversight → Usage analytics → Quality control

Include Turkish educational context, school building aesthetics, and modern library technology integration. Background: School campus environment with digital overlays.
```

### 8. Çok-Kullanıcılı Sistem Mimarisi

**Prompt:**

```
Design a multi-user system architecture for intelligent library:

- Layout: Layered architecture with user role separation
- User layers:
  * Student Interface: Mobile apps, web portal, kiosk terminals
  * Teacher Portal: Content management, class monitoring, analytics
  * Librarian Console: System administration, content curation, reports
  * Parent Access: Student progress, resource recommendations

- Core system layers:
  * Authentication & Authorization: Role-based access control
  * Content Management: Multi-tenant knowledge bases per course
  * RAG Processing: Subject-aware processing pipelines
  * Analytics Engine: Usage tracking, learning insights, performance metrics

Include Turkish education system hierarchy, school organizational structure, and multi-language support indicators. Style: Enterprise architecture diagram with educational institution branding.
```

## 🗂️ Bilgi Havuzu Yönetimi Görselleri

### 9. Ders-Özel Bilgi Havuzu Organizasyonu

**Prompt:**

```
Create a visual representation of course-specific knowledge pool organization:

- Layout: Hierarchical tree structure with educational taxonomy
- Organization levels:
  * School Level: Institution branding and overall structure
  * Department Level: Subject departments (Matematik, Türkçe, Fen, Sosyal)
  * Grade Level: Class levels (5. sınıf, 6. sınıf, etc.)
  * Course Level: Specific courses (Matematik 8A, Türkçe 9B)
  * Unit Level: Curriculum units (Cebirsel İfadeler, Ottoman History)
  * Resource Level: Individual documents and materials

Visual elements:
- Folder icons with subject-specific colors and symbols
- Document type icons (PDF, DOCX, PPTX, video, audio)
- Access control indicators (public, class-only, teacher-only)
- Usage statistics (most accessed, recently added, trending)
- Cross-references between related topics

Include Turkish curriculum structure, realistic course names, and educational content examples. Style: File system visualization with educational design elements.
```

### 10. Öğretmen Colaborasyonu ve İçerik Paylaşımı

**Prompt:**

```
Design a collaboration interface for teachers to share and build upon each other's knowledge pools:

- Layout: Social learning network with professional networking elements
- Features:
  * Teacher profiles with subject expertise and teaching experience
  * Shared content library with peer ratings and reviews
  * Collaboration requests and content exchange
  * Best practices sharing and success stories
  * Cross-department resource sharing

Workflow visualization:
- Content discovery: Browse other teachers' approved materials
- Permission requests: Ask to use or adapt another teacher's content
- Collaboration tools: Co-create resources with other educators
- Quality assurance: Peer review and validation system
- Attribution system: Proper crediting for shared resources

Include Turkish educational culture elements, teacher professional development aspects, and realistic collaboration scenarios. Style: Professional social network interface with educational focus, similar to LinkedIn but for educators.
```

## 🖥️ UI/UX Mockup Görselleri

### 11. Öğrenci Chatbot Arayüzü

**Prompt:**

```
Create a student-friendly chatbot interface for the intelligent library system:

- Layout: Modern chat interface optimized for educational conversations
- Age-appropriate design elements for different grade levels
- Color scheme: Student-friendly (Bright blue: #2196F3, Encouraging green: #4CAF50, Warning orange: #FF9800)
- Features:
  * Subject selection dropdown (hangi ders için soru soruyorsun?)
  * Voice input option for accessibility
  * Visual elements like emojis and illustrations
  * Progress indicators showing learning journey
  * Quick suggestion buttons for common question types
  * Source citation in student-friendly format

Chat examples:
- Math questions with step-by-step solutions
- Literature analysis with relevant text excerpts
- Science concepts with visual aids and examples
- History questions with timeline context

Include Turkish student language patterns, educational encouragement phrases, and age-appropriate interaction design. Style: Friendly educational app similar to Duolingo or Khan Academy mobile apps.
```

### 12. Akıllı Kütüphane Mobil Uygulaması

**Prompt:**

```
Design a comprehensive mobile app for the intelligent library system:

- Multi-user interface supporting students, teachers, and parents
- Bottom navigation with role-specific tabs
- Student view: Soru Sor, Derslerim, Geçmiş, Profil
- Teacher view: İçerik Yönet, Öğrenciler, Analitik, Ayarlar
- Parent view: Çocuğum, İlerleme, Raporlar, İletişim

Screen designs:
- Login/role selection screen with school branding
- Course selection with visual subject icons
- Chat interface with multimedia support
- Content upload with progress indicators
- Analytics dashboard with charts and insights
- Settings with privacy and notification controls

Include Turkish localization, offline capability indicators, accessibility features, and responsive design for various screen sizes. Style: Modern educational app with consistent design system and intuitive navigation.
```

### 6. Mobil Responsive Tasarım

**Prompt:**

```
Design mobile-responsive views of the RAG system interface:

- Show 3 device views: Desktop (1920px), Tablet (768px), Mobile (375px)
- Consistent branding across all sizes
- Mobile-first design principles
- Touch-friendly interface elements
- Collapsible navigation
- Optimized text input for mobile keyboards
- Swipe gestures for source browsing

Style: Modern mobile UI with card-based layout, plenty of whitespace, clear typography. Include realistic Turkish educational content and user interactions.
```

## 📚 Eğitici İnfografik Görselleri

### 7. RAG Nedir? Açıklama İnfoğrafi

**Prompt:**

```
Create an educational infographic explaining "What is RAG?" in Turkish:

- Layout: Vertical infographic suitable for social media sharing
- Target audience: University students and educators
- Content sections:
  1. "RAG Nedir?" - Simple definition with icons
  2. "Geleneksel AI vs RAG" - Comparison visual
  3. "RAG'ın Avantajları" - Benefits with icons
  4. "Eğitimde RAG Kullanımı" - Educational use cases
  5. "Sistem Bileşenleri" - Simple architecture overview

Style: Modern infographic design with illustrations, consistent color scheme (educational blues and oranges), clear hierarchy, engaging visuals. Include statistics and real-world examples.
```

### 8. Embedding Kavramı Görselleştirmesi

**Prompt:**

```
Design an educational visualization explaining text embeddings:

- Concept: Show how text is converted to mathematical vectors
- Visual elements:
  * Text samples in Turkish ("Yapay zeka", "Machine learning", "Derin öğrenme")
  * Arrow transformations
  * Vector representations (colored dots in 3D space)
  * Similarity clustering visualization
  * Distance measurements between similar concepts

Style: Clean, scientific visualization with bright colors, clear labels in Turkish. Similar to educational materials used in computer science courses. Include step-by-step explanation.
```

## 🔧 Teknik Diyagram Görselleri

### 9. Vektör Veritabanı Mimarisi

**Prompt:**

```
Create a technical diagram showing vector database architecture:

- Focus on FAISS implementation
- Show components:
  * Index structures (Flat, IVF)
  * Memory mapping
  * Similarity search algorithms
  * Clustering visualization
  * Performance characteristics

Style: Technical blueprint with engineering aesthetics, blue and white color scheme, precise geometric shapes, detailed annotations in Turkish. Include performance metrics and scalability indicators.
```

### 10. Ollama Entegrasyonu Diyagramı

**Prompt:**

```
Design a system integration diagram showing Ollama integration:

- Show the complete flow:
  * Local Ollama server
  * Model loading (Llama 3, mxbai-embed-large)
  * API communication
  * Request/response handling
  * Error handling and fallbacks

Include network topology, API endpoints, data formats. Style: Network diagram with server icons, clean lines, technical color scheme (grays, blues, greens). Add performance metrics and system requirements.
```

## 📈 Sunum ve Dokümantasyon Görselleri

### 11. Proje Poster Tasarımı

**Prompt:**

```
Create an academic conference poster for the RAG education project:

- Size: A1 portrait format (594 x 841 mm)
- Sections:
  * Header: Project title, university, author, advisor
  * Abstract/Özet
  * System Architecture
  * Key Features
  * Results/Sonuçlar
  * Future Work/Gelecek Çalışmalar
  * References/Kaynaklar

- Style: Academic poster with professional layout
- Colors: University branding colors
- Include charts, diagrams, screenshots
- QR code for demo access
- Clean typography, good whitespace usage

Design should be suitable for academic conferences and university presentations.
```

### 12. Başarı Metrikleri Dashboard

**Prompt:**

```
Design a metrics dashboard showing system performance:

- Layout: Executive dashboard style
- Key metrics displayed:
  * Query response time charts
  * User satisfaction scores
  * Document processing statistics
  * System uptime indicators
  * Usage analytics graphs

Style: Modern business intelligence dashboard with clean charts, consistent color coding, interactive elements suggested. Include Turkish labels and realistic data visualization.
```

## 💡 Prompt Optimizasyon İpuçları

### Genel Kurallar:

1. **Spesifik olun**: Renk kodları, boyutlar ve stil referansları ekleyin
2. **Bağlam verin**: Projenin eğitim odaklı olduğunu belirtin
3. **Dil belirtin**: Türkçe içerik istediğinizi açık şekilde belirtin
4. **Stil referansları**: Bilinen markalar veya tasarım stilleri referans alın
5. **Teknik detaylar**: Çözünürlük, format ve kullanım amacını belirtin

### Her Model İçin Özelleştirme:

**DALL-E 3 için:**

- Uzun, detaylı promptlar kullanın
- Negatif promptlara gerek yok
- Tekrar ve vurgu yapın

**Midjourney için:**

- `--ar` aspect ratio parametrelerini kullanın
- `--style raw` teknik diyagramlar için
- `--v 6` son versiyonu belirtin

**Stable Diffusion için:**

- Negatif promptlar ekleyin: `blurry, low quality, distorted`
- ControlNet kullanarak şekil kontrolü yapın
- LoRA modelleri ile özelleştirin

### Yaygın Hatalar:

❌ Belirsiz renkler ("mavi" yerine hex kodu)
❌ Genel açıklamalar ("güzel görünsün")
❌ Tek seferde çok fazla element
❌ Bağlam eksikliği
❌ Dil karışıklığı

✅ Spesifik renk kodları (#2E86AB)
✅ Detaylı stil açıklaması
✅ Odaklanılmış içerik
✅ Proje bağlamı
✅ Consistent Türkçe kullanımı

---

## 🎨 Prompt Şablonları

### Sistem Diyagramı Şablonu:

```
Create a [diagram_type] showing [system_name] with:
- Style: [visual_style]
- Colors: [color_scheme]
- Components: [list_components]
- Layout: [arrangement_description]
- Labels: Turkish
- Background: [background_description]
- Quality: Professional, technical documentation style
```

### UI Mockup Şablonu:

```
Design a [interface_type] for [application_name]:
- Platform: [web/mobile/desktop]
- Dimensions: [specific_size]
- Color scheme: [brand_colors]
- Layout sections: [list_sections]
- Content: Realistic Turkish educational content
- Style: Modern, accessible, [reference_style]
- Include: [specific_elements]
```

### İnfografik Şablonu:

```
Create an educational infographic about [topic]:
- Target audience: [user_type]
- Language: Turkish
- Layout: [vertical/horizontal/grid]
- Sections: [content_structure]
- Style: [design_approach]
- Colors: [color_palette]
- Include: Statistics, examples, clear hierarchy
```

---

## 📞 Destek ve Kaynaklar

Bu prompt rehberini kullanırken karşılaştığınız sorunlar için:

1. **Prompt mühendisliği**: Her model için özel optimizasyonlar
2. **Stil tutarlılığı**: Tüm görseller için consistent branding
3. **Teknik doğruluk**: Mimari diyagramların sisteme uygunluğu
4. **Eğitici değer**: Görselların öğretici içerik sunması

**Not:** Bu rehber, RAG3 projesi özelinde tasarlanmış olup, her prompt projenizin spesifik ihtiyaçlarına göre özelleştirilebilir.

---

_Bu dokümantasyon, eğitim amaçlı RAG sistemi projesi için hazırlanmıştır. Tüm promptlar Türkçe eğitim içeriği ve akademik sunum kalitesi gözetilerek optimize edilmiştir._

## 📈 Sunum ve Dokümantasyon Görselleri

### 16. Akıllı Kütüphane Projesi Akademik Poster

**Prompt:**

```
Create an academic conference poster for the Intelligent Library RAG project:

- Size: A1 portrait format (594 x 841 mm)
- Title: "Eğitim Alanında RAG Kullanımı ile Akıllı Kütüphane Sistemi: Öğretmen Odaklı Bilgi Havuzu Yaklaşımı"
- Sections:
  * Header: Project title, university logo, education faculty branding
  * Problem Statement: Traditional library limitations in digital education
  * Solution Overview: Teacher-driven knowledge pool creation with RAG
  * System Architecture: Multi-user system with teacher, student, librarian roles
  * Key Innovations: Turkish language support, curriculum alignment, teacher oversight
  * Implementation Results: Teacher adoption rates, student engagement metrics, learning outcomes
  * Case Studies: Real classroom implementations across different subjects
  * Future Developments: AI tutoring integration, multi-school deployment
  * Acknowledgments: Education ministry collaboration, participating schools

Visual elements:
- Before/after comparison of library usage
- Teacher workflow diagrams
- Student interaction screenshots
- Performance metrics with Turkish educational context
- QR codes for live demo and video presentation

Style: Academic education conference poster with modern design, institutional colors, clear visual hierarchy, and education-focused imagery.
```

### 17. Öğretmen Başarı Metrikleri ve Analitik Dashboard

**Prompt:**

```
Design a comprehensive analytics dashboard for teachers and administrators:

- Layout: Multi-panel dashboard with role-based views
- Teacher Metrics Panel:
  * Content creation statistics (uploaded documents, approved materials)
  * Student engagement rates (questions per week, active learners)
  * Subject-specific usage (most popular topics, difficult concepts identified)
  * Knowledge pool effectiveness (answer accuracy, student satisfaction)

- Administrator Overview:
  * School-wide adoption rates across departments
  * Teacher participation and training progress
  * Student learning outcome improvements
  * System performance and resource utilization

- Student Learning Analytics:
  * Question patterns and learning paths
  * Subject difficulty analysis
  * Progress tracking and achievement metrics
  * Peer comparison and collaborative learning indicators

Visual elements:
- Interactive charts showing Turkish educational calendar alignment
- Heat maps of content usage by time and subject
- Trend lines showing learning progression
- Geographic maps for multi-school implementations
- Alert systems for content quality and student support needs

Style: Modern education analytics platform with clean data visualization, Turkish interface, and education-specific KPIs. Color coding for different user roles and urgency levels.
```

### 18. Kurumsal Sunum Template (Okul Yönetimine)

**Prompt:**

```
Create a presentation template for pitching the intelligent library system to school administrators:

- Format: PowerPoint-style slides optimized for institutional presentations
- Slide types needed:
  * Title slide with school branding compatibility
  * Problem statement (current library limitations, digital transformation needs)
  * Solution overview (intelligent library concept with teacher empowerment)
  * ROI analysis (cost savings, efficiency gains, learning improvements)
  * Implementation timeline (pilot program to full deployment)
  * Success stories (testimonials from participating teachers and students)
  * Technical requirements (infrastructure, training, support needs)
  * Budget breakdown (initial setup, ongoing maintenance, expansion costs)

Design elements:
- Professional education sector aesthetic
- Consistent color scheme adaptable to different school brandings
- Clean typography suitable for projection
- Infographic elements for complex concepts
- Space for school-specific customization
- Charts and graphs templates for data presentation

Include Turkish educational terminology, ministry compliance indicators, and realistic implementation scenarios for different school sizes and types.
```

---

## 🎯 Özel Prompt Önerileri

### Öğretmen Odaklı Görsel İçerik:

- **Ders planı entegrasyonu**: Öğretmenlerin mevcut ders planlarına nasıl entegre edeceklerini gösteren görseller
- **Subject-specific workflows**: Her ders için özelleştirilmiş iş akışları (Matematik, Türkçe, Fen, Sosyal Bilgiler)
- **Sınıf yönetimi**: Öğrenci grupları ve sınıf seviyelerine göre içerik organizasyonu

### Kurumsal Sunum Materyalleri:

- **Bütçe justification**: Yatırım getirisini gösteren finansal görseller
- **Training materials**: Öğretmen eğitimi için görsel rehberler
- **Success metrics**: Başarı ölçütleri ve benchmark karşılaştırmaları

### Teknik Dokümantasyon:

- **System requirements**: Kurulum ve sistem gereksinimleri
- **Security compliance**: Veri güvenliği ve gizlilik standartları
- **Integration guides**: Mevcut okul sistemleriyle entegrasyon

**Not:** Bu rehber, Akıllı Kütüphane RAG sistemi projesi özelinde tasarlanmış olup, her prompt eğitim kurumlarının spesifik ihtiyaçlarına göre özelleştirilebilir.

---

_Bu dokümantasyon, eğitim kurumları için akıllı kütüphane RAG sistemi projesi kapsamında hazırlanmıştır. Tüm promptlar Türkçe eğitim içeriği, öğretmen iş akışları ve kurumsal sunum kalitesi gözetilerek optimize edilmiştir. Proje, öğretmenlerin kendi derslerine özel bilgi havuzları oluşturarak chatbot tarzı eğitim asistanları kurmasını hedeflemektedir._
