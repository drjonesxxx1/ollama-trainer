import { MCPToolDeclaration } from "../types";

export const DEFAULT_MCP_PRESETS: MCPToolDeclaration[] = [
  {
    id: "filesystem-mcp",
    name: "read_file",
    serverName: "Filesystem MCP",
    description: "Read the full contents of a file from the user's workspace securely.",
    parametersSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "The relative or absolute file path to read" },
        start_line: { type: "number", description: "Optional starting line index (1-based)" },
        end_line: { type: "number", description: "Optional ending line index" },
      },
      required: ["path"],
    },
    sampleCallsCount: 45,
  },
  {
    id: "filesystem-write",
    name: "write_file",
    serverName: "Filesystem MCP",
    description: "Create or overwrite a file with given text content.",
    parametersSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "Path where the file should be created" },
        content: { type: "string", description: "The complete content to write" },
      },
      required: ["path", "content"],
    },
    sampleCallsCount: 38,
  },
  {
    id: "postgres-query",
    name: "execute_sql",
    serverName: "PostgreSQL MCP",
    description: "Execute a read-only or transactional SQL query against the connected database.",
    parametersSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Valid PostgreSQL query statement" },
        limit: { type: "number", description: "Maximum rows to return" },
      },
      required: ["query"],
    },
    sampleCallsCount: 52,
  },
  {
    id: "websearch-mcp",
    name: "web_search",
    serverName: "Web Search MCP",
    description: "Perform real-time search across the web and return top synthesized results.",
    parametersSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "The search query string" },
        num_results: { type: "number", description: "Number of search results to fetch (1-10)" },
      },
      required: ["query"],
    },
    sampleCallsCount: 60,
  },
  {
    id: "github-mcp",
    name: "create_pull_request",
    serverName: "GitHub MCP",
    description: "Create a new pull request on a GitHub repository with title and branch details.",
    parametersSchema: {
      type: "object",
      properties: {
        repo: { type: "string", description: "owner/repository_name format" },
        title: { type: "string", description: "Pull request title" },
        head_branch: { type: "string", description: "The source feature branch" },
        base_branch: { type: "string", description: "The target branch (e.g. main)" },
        body: { type: "string", description: "PR description in markdown" },
      },
      required: ["repo", "title", "head_branch", "base_branch"],
    },
    sampleCallsCount: 29,
  },
];
