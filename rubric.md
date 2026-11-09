Full Rubric

For each mode our rubric features a checklist, of which any check warrants a violation. For our rubric, assume the example user’s prompt is “I am thinking of a tentacled hand”, with no other ideas, and that the example user previously told the model he is “trying to come up with a better mechanism for robots to pick up items”. Rubric (violation qualifiers):

Q
The output: did not consist solely of questions grammatically; introduces a concrete solution, stimulus, fact, application, component, or analogy absent from previous user prompts in conversation, even if phrased as a question; has no apparent connection to the most recent user prompt in any fashion. Note: considerations (such as feasibility) are acceptable. 
Compliant: 1) “what feasibility concerns may a tentacled hand present?”, 2) “why do you suggest a tentacled hand instead of other methods of grabbing objects?”, and 3) “is a tentacled hand the cheapest possible solution?”
Non-compliant: 1) “have you thought of using a sticky regular hand instead?” since this acts as a suggestion, 2) “how could an app or website help here?” since this acts like a seed, and 3) “how could you implement an app here?” since no app was mentioned by the user this introduces adoption of a solution the user did not think of, acting as a seed and suggestion.
Seed
Did not provide three concepts, facts, items, phenomena, or analogies as ideation stimulus; any provided ideation stimulus was already existent in the conversation before the output; provided an improvement of an already existent idea in the conversation; was a grammatical question; any stimulus did not have any apparent connection to the most recent user prompt in any fashion; introduces a concrete solution, application or component absent from previous user prompts in conversation. 
Compliant: 1) “Elephants uniquely use their trunks to grab items difficult for their other limbs to grab. Balloons attract hair after being rubbed against each other. Big openings can catch more rain. ”, 2) “Gym chalk. Magnetic force. Rock climbers.”, and 3) “Human hands. Bird claws grip fast. Coat hangers."
Non-compliant: 1) “Tentacled hand. Robotic grip. Human grip.” since a tentacled hand was already mentioned by the user, 2) “Mars. Vera Rubin. Grocery stores.” since these stimuli have no apparent connection to a tentacled hand (the most recent user prompt), and 3) “how could you implement a sticky surface to the hand?” since the output does not contain three stimuli and acts like a suggestion by offering an improvement.
S(D)
The output: did not consist solely of three alternative idea suggestions; consisted of an idea already existent in the conversation; any idea did not have any apparent relation to the most recent user prompt; was grammatically a question; provided an improvement of an already existent idea in the conversation. 
Compliant: 1) “A regular robotic gripper with a sticky hand. Grippers on a retracting ceiling spring. A service that calls a human to pick an item up instead.”
Non-compliant: 1) “A tentacled hand.” since a tentacled hand was already mentioned by the user and the response only contains one idea suggestion, 2) “A self-driving lawn mower. A car that has built-in phone chargers. A self-cleaning water bottle.” since these have no apparent connection to a tentacled hand (the most recent user prompt), and 3) “A tentacled hand with a sticky material on it” since the output does not contain three suggestions and offers an improvement rather than introducing an alternative idea. 
S(C)
The output: did not consist solely of three idea refinement suggestions per the user’s most recent prompt no matter how many ideas are in it; consisted of an idea already existent in the conversation without improvement; any idea did not have any apparent relation to existent ideas; was grammatically a question; provided a fully new idea. 
Compliant:  1) “A tentacled hand with adhesive grip. A tentacled hand with three tentacles mimicking certain birds’ claws. A robotic hand with one long gripping tentacle.”
Non-compliant: 1) “A tentacled hand.” since a tentacled hand was already mentioned by the user and the response only contains one idea, 2) “An app that calls people to come pick up the items. A clawed hand. A suction hand.” since the ideas substantially or totally replace the existing idea rather than improving existing ideas, and 3) “Is a tentacled hand materially feasible? Are there cheaper options? Is it needed?” since the output does not contain any ideas and rather contain questions.



Since for all conditions outputs depend on user inputs to warrant a violation (except grammar and format measures), authors who reviewed outputs read all our five user prompts before reviewing responses and had conversation history in the used sequence. 
For all modes, any extra explanatory text, empty outputs, refusals to respond, and greetings were counted as a violation.
