# Expert Knowledge Twin
## Executive Summary
Students and early-career professionals frequently face high-stakes academic and career decisions, yet
access to reliable and personalized mentorship remains limited. Traditional one-to-one mentorship is often
expensive, time-consuming, and difficult to access, while scalable alternatives such as online forums or
generic AI tools fail to deliver advice that is context-aware and grounded in real-world experience. As
a result, users rely on sources they do not fully trust when making decisions that can have long-term
personal and professional consequences. This problem is compounded by the fact that many experienced
professionals and creators possess valuable domain knowledge but lack efficient ways to share and monetize
it at scale without significant ongoing effort. This creates a mismatch between demand for trusted guidance
and the availability of accessible expert input. To address this gap, we built Xpert, a fully functional web
application that enables users to access AI-powered expert twins trained on the real-world experience
of verified experts. The current MVP focuses on the career and education domain and allows users to
browse expert profiles, explore their social presence and background, interact with an AI twin via chat,
and provide ratings and reviews. The product was shaped and evaluated through multiple rounds of
research, including:
- 41 survey respondents exploring pain points, value, and willingness to pay
- 4 creator interviews conducted to validate expert-side value
- 9 qualitative interviews conducted regarding problem formation
- 6 users participating in an MVP evaluation while completing predefined tasks using a think-aloud
protocol

Research consistently highlighted two key findings: cost and accessibility are the primary barriers to
traditional mentorship, and trust is a non-negotiable requirement for AI-generated advice. Users expressed
significantly higher confidence when advice was clearly linked to a real, verified expert and supported by
visible human oversight. The research and MVP validation indicate strong user interest in expert-driven,
AI-supported guidance when trust conditions are met. The next recommended step for us is to improve AI
response quality and further align the AI twin’s outputs with the expert’s authentic voice and reasoning.
Strengthening answer quality and nuance is critical to reinforcing trust and differentiating Xpert from
generic AI tools as the platform scales.

## Code Structure
Folder Name | Description 
--- | --- 
n8n_workflows | contains all the n8n workflows that we have in order to support our system.
supabase | contains supabase migration script
aimodel | contains llm hosting code and useful commands. llamaMistral3host.sh is the main script to run the llm. However, environment and llama setup might be needed before running the script. Please check useful_commands.txt and loadllama.sh for the setup.
embeddingModel | contains embedding model hosting commands. hostMistralEmbedding.sh is the main script to run the embedding model.
