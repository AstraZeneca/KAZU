We now assign a MentionConfidence to each Entity based on the confidence of the NER hit.
This allows decoupling between specific NER steps and disambiguation strategies, which were previously intertwined.
This also applies to the CleanupStep which is decoupled from specific NER steps.
