# AI Document Summarizer

An advanced document summarization system that leverages LLMs and vector indexing to generate comprehensive summaries of multiple documents simultaneously.

## Features

### 1. Multi-Model Support
- OpenAI GPT-4 Turbo integration
-Google Gemini 1.5 Flash support

### 2. Document Processing
- LlamaParse integration for parsing multiple document formats
- Vector indexing using LlamaIndex
- Chunk-based processing with configurable sizes

### 3. Summary Generation Pipeline

#### Document Parsing & Vectorization
- Automatic document chunking with SentenceSplitter
- Vector store indexing for efficient retrieval
- Embedding model selection based on LLM choice

#### Summary Generation
- Query-based summary extraction
- Key points identification
- Important information preservation

#### Summary Formatting
- AI agent-based formatting system
- Structured output with clear headings
- Readability optimization

### 4. Batch Processing
- Multiple file upload support
- Asynchronous processing
- Progress tracking


## Technical Architecture

### 1. Core Components

#### Vector Store Setup
- Chunk size: 512 tokens
- Custom embedding models
- Query engine configuration

#### LLM Integration
- Temperature: 0.6
- Max tokens: 2000
- Async initialization
- Error handling

#### Formatting Agent
- Role: Summary Formatter
- Goal: Clear and concise structuring
- Single-task focused
- No delegation allowed

### 2. Processing Pipeline

1. **Document Intake**
   - File upload handling
   - Format validation
   - Temporary storage

2. **Processing**
   - Document parsing
   - Vector indexing
   - Query execution
   - Summary generation

3. **Post-Processing**
   - AI formatting
   - Structure addition
   - Quality verification

4. **Output Generation**
   - Table display
   - Word document compilation
   - Download preparation

## Features

### 1. Input Support
- Multiple file upload
- Various document formats
- Batch processing

### 2. Processing
- Asynchronous execution
- Progress tracking
- Error handling

### 3. Output Options
- Interactive table view
- Downloadable Word document
- Structured formatting

## Performance Considerations

1. **Memory Management**
   - Chunk-based processing
   - Temporary file handling
   - Buffer cleanup

2. **Processing Optimization**
   - Async operations
   - Batch processing
   - Resource cleanup

3. **Output Handling**
   - Streaming responses
   - Memory-efficient document generation
   - Progressive loading

## Limitations

1. **API Dependencies**
   - Requires valid API keys for:
     - OpenAI or Google Gemini
     - LlamaParse
   - Rate limiting considerations

2. **Resource Requirements**
   - Memory usage scales with document size
   - Processing time varies with complexity
   - Storage needs for temporary files

3. **Format Support**
   - Limited by LlamaParse capabilities
   - Size restrictions may apply
   - Some formats may have reduced parsing quality

## Error Handling

1. **API Validation**
   - Key verification
   - Service availability checks
   - Rate limit management

2. **Processing Errors**
   - Document parsing failures
   - Summary generation issues
   - Formatting problems

3. **Output Errors**
   - Document generation failures
   - Download issues
   - Display problems

## Dependencies
