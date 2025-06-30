def run_graph_rag(question):
    global driver
    if driver is None:
        print("Driver not available. Cannot connect to graph.")
        return
    with driver.session(database="neo4j") as session:
        print("step 1 : Retrieving graph schema ...")
        schema = driver.execute_read(get_graph_scheam)
        print("step 2 : translating question to chuper query using llm...")
        cypher_query = question_to_cypher(question,schema)
        print(f"   - Generated Cypher: {cypher_query}")

        print("Step 3: Executing query and retrieving data from the graph...")
        try:
            result = session.run(cypher_query).data()
            if not result:
                print("   - No data returned from the graph.")
                context = "The knowledge graph did not contain information for this query."
            else:
                context = json.dumps(result, indent=2)
                print(f"   - Retrieved {len(result)} results.")
        except Exception as e:
            print(f"   - Error executing Cypher query: {e}")
            context = "There was an error querying the knowledge graph."
        print("Step 4: Generating final answer using the LLM...")
        final_answer = genrate_final_answer(question,context)
        print("\\n" + "="*50)
        print(" KNOWLEDGE GRAPH RESPONSE ".center(50, "="))
        print("="*50)
        print(f"\\nQuestion: {question}\\n")
        print(f"Answer: {final_answer}")
        print("\\n" + "="*50)
