;;; EU AI Act - Art. 50 Transparency Obligations (Limited Risk)
;;; Salience 60

(defrule limited-chatbot-disclosure
  "Art 50(1): Chatbots must disclose they are not human"
  (declare (salience 60))
  (ai-system (is-chatbot           ?cb&:(> ?cb 0.4))
             (interacts-with-people ?ip&:(> ?ip 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-ART50-1")
    (risk-tier  "LIMITED")
    (article    "Art. 50(1)")
    (reason     "Conversational AI / chatbot that interacts with natural persons must inform users they are interacting with an AI (unless obvious from context)")
    (confidence (min-float ?cb ?ip))
  ))
)

(defrule limited-emotion-recognition-disclosure
  "Art 50(3): Emotion recognition systems must inform subjects"
  (declare (salience 60))
  (ai-system (emotion-recognition  ?er&:(> ?er 0.4))
             (interacts-with-people ?ip&:(> ?ip 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-ART50-3")
    (risk-tier  "LIMITED")
    (article    "Art. 50(3)")
    (reason     "Emotion recognition system must inform natural persons when they are being subjected to it")
    (confidence (min-float ?er ?ip))
  ))
)

(defrule limited-synthetic-content-labelling
  "Art 50(2)/(4): Synthetic content (deepfakes) must be labelled"
  (declare (salience 60))
  (ai-system (generates-synthetic ?gs&:(> ?gs 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-ART50-4")
    (risk-tier  "LIMITED")
    (article    "Art. 50(2)/(4)")
    (reason     "AI-generated or manipulated image, audio, or video content (deepfakes) must be disclosed as artificially generated")
    (confidence ?gs)
  ))
)
