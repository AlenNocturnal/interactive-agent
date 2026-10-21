import sys
from smolagents import CodeAgent, LiteLLMModel

def main():
    print("🤖 Initializing your local agent...")
    
    # 1. Connect to your local Ollama runtime using Qwen 2.5 3B
    try:
        model = LiteLLMModel(
            model_id="ollama/qwen2.5:3b",
            api_base="http://localhost:11434"
        )
    except Exception as e:
        print(f"\n❌ Error connecting to Ollama: {e}")
        print("👉 Please verify that Ollama is running (`ollama serve`) and the model is downloaded (`ollama pull qwen2.5:3b`).")
        sys.exit(1)

    # 2. Setup the autonomous CodeAgent
    agent = CodeAgent(
        tools=[], 
        model=model,
        add_base_tools=True  # Gives the agent baseline logic toolsets
    )

    print("\n=======================================================")
    print("🤖 Local Interactive Agent is Ready!")
    print("   Type your task or question below.")
    print("   Type 'exit' or 'quit' to end the session.")
    print("=======================================================\n")

    # 3. Interactive loop
    while True:
        try:
            # Capture user prompt
            user_input = input("👤 You: ").strip()
            
            # Check for exit commands
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye!")
                break
                
            print("\n⚙️ Agent is thinking and executing actions...")
            
            # Run the agent execution loop for the current instruction
            # smolagents maintains an internal state across turns within its execution context
            response = agent.run(user_input)
            
            print("\n=======================================================")
            print(f"🎯 Agent Final Answer:\n{response}")
            print("=======================================================\n")
            
        except KeyboardInterrupt:
            # Handle Ctrl+C cleanly
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred during execution: {e}\n")

if __name__ == "__main__":
    main()

