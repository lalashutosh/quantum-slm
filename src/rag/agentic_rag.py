from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Import your existing V1 components
from .rag_system import QuantumRAG

class AgenticQuantumRAG(QuantumRAG):
    """Upgraded Orchestrator featuring Agentic RAG capabilities."""

    def __init__(self, persist_path: str, embedding_model: str, model, tokenizer):
        # Initialize V1 components
        super().__init__(persist_path, embedding_model)
        
        # 1. Wrap your local model in a HuggingFace Pipeline for LangChain compatibility
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=250,
            temperature=0.1,
            repetition_penalty=1.1,
            top_p=0.9,
            do_sample=True,
            return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
        # 2. Set up the Agent Executor
        self.agent_executor = self._setup_agent()

    def _setup_agent(self):
        """Creates the Agent and defines its tools."""
        
        # Define the retriever as a tool the agent can call
        tools = [
            Tool(
                name="Quantum_Knowledge_Base",
                func=self._tool_retrieve,
                description="Use this tool to search for quantum physics papers, equations, and mathematical contexts. Input should be a specific search query."
            )
            # You can easily add more tools here later:
            # Tool(name="Calculator", func=math_eval, description="Use for solving equations.")
        ]

        # Define a ReAct (Reason + Act) prompt
        # This teaches the LLM to think step-by-step and use tools
        react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(react_template)

        # Create the agent and the executor
        agent = create_react_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, # Great for debugging the agent's "Thought" process
            handle_parsing_errors=True,
            max_iterations=4 # Prevents infinite loops if the model gets confused
        )

    def _tool_retrieve(self, query: str) -> str:
        """Helper method to format retrieved docs for the agent."""
        # Using your V1 MMR retrieval
        context_docs = self.retriever.mmr_retrieve(query)
        if not context_docs:
            return "No relevant documents found in the knowledge base."
        return "\n\n".join([d.page_content for d in context_docs])

    def generate_agentic_response(self, query: str):
        """Executes the agentic RAG loop."""
        print(f"🤖 Agent is thinking about: {query}")
        result = self.agent_executor.invoke({"input": query})
        return result['output']