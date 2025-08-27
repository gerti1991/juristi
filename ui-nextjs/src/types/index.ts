// API Types
export interface SearchRequest {
  query: string;
  top_k?: number;
  mode?: string;
  rag_mode?: string;
}

export interface DocumentResult {
  id: string;
  title: string;
  content: string;
  source: string;
  similarity_score: number;
}

export interface SearchResponse {
  success: boolean;
  query: string;
  results: DocumentResult[];
  total_results: number;
  search_time: number;
}

export interface AnalyseRequest {
  query: string;
}

export interface AnalyseResponse {
  success: boolean;
  query: string;
  response: string;
  analysis_time: number;
}

// UI Types
export interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  mode?: 'precise' | 'analyzed';
  sources?: DocumentResult[];
  loading?: boolean;
}

export interface AppState {
  messages: Message[];
  currentQuery: string;
  isLoading: boolean;
  error: string | null;
}
