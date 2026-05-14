;;; EU AI Act - Art. 5 Prohibited AI Practices
;;; Salience 100: these must fire before all other rules

(defrule unacceptable-social-scoring
  "Art 5(1)(c): Social scoring of natural persons by public authorities"
  (declare (salience 100))
  (ai-system (social-scoring   ?ss&:(> ?ss 0.4))
             (public-authority ?pa&:(> ?pa 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-ART5-C")
    (risk-tier  "UNACCEPTABLE")
    (article    "Art. 5(1)(c)")
    (reason     "Social scoring of natural persons by a public authority based on their behaviour or personal characteristics")
    (confidence (min-float ?ss ?pa))
  ))
)

(defrule unacceptable-rtbi-public-space
  "Art 5(1)(h): Real-time remote biometric identification in publicly accessible spaces"
  (declare (salience 100))
  (ai-system (biometric-realtime ?br&:(> ?br 0.4))
             (public-space       ?ps&:(> ?ps 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-ART5-H")
    (risk-tier  "UNACCEPTABLE")
    (article    "Art. 5(1)(h)")
    (reason     "Real-time remote biometric identification of natural persons in publicly accessible spaces")
    (confidence (min-float ?br ?ps))
  ))
)

(defrule unacceptable-subliminal-manipulation
  "Art 5(1)(a): Subliminal techniques causing harm"
  (declare (salience 100))
  (ai-system (subliminal-manipulation ?sm&:(> ?sm 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-ART5-A")
    (risk-tier  "UNACCEPTABLE")
    (article    "Art. 5(1)(a)")
    (reason     "AI system deploying subliminal techniques beyond consciousness to materially distort behaviour in a manner causing harm")
    (confidence ?sm)
  ))
)

(defrule unacceptable-vulnerable-exploitation
  "Art 5(1)(b): Exploiting vulnerabilities of specific groups"
  (declare (salience 100))
  (ai-system (subliminal-manipulation ?sm&:(> ?sm 0.4))
             (targets-vulnerable      ?tv&:(> ?tv 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-ART5-B")
    (risk-tier  "UNACCEPTABLE")
    (article    "Art. 5(1)(b)")
    (reason     "AI system exploiting vulnerabilities of specific groups (age, disability, social situation) causing harm")
    (confidence (min-float ?sm ?tv))
  ))
)

(defrule unacceptable-predictive-policing
  "Art 5(1)(d): Individual risk assessments for predictive policing based solely on profiling"
  (declare (salience 100))
  (ai-system (predictive-policing ?pp&:(> ?pp 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-ART5-D")
    (risk-tier  "UNACCEPTABLE")
    (article    "Art. 5(1)(d)")
    (reason     "AI system making individual risk assessments for predictive policing based solely on profiling")
    (confidence ?pp)
  ))
)

(defrule unacceptable-emotion-workplace
  "Art 5(1)(f): Emotion recognition in the workplace"
  (declare (salience 100))
  (ai-system (emotion-recognition ?er&:(> ?er 0.4))
             (workplace-use       ?wu&:(> ?wu 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-ART5-F-WORK")
    (risk-tier  "UNACCEPTABLE")
    (article    "Art. 5(1)(f)")
    (reason     "Emotion recognition system deployed in the workplace (with exceptions for safety purposes)")
    (confidence (min-float ?er ?wu))
  ))
)

(defrule unacceptable-emotion-education
  "Art 5(1)(f): Emotion recognition in educational institutions"
  (declare (salience 100))
  (ai-system (emotion-recognition ?er&:(> ?er 0.4))
             (education-use       ?eu&:(> ?eu 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-ART5-F-EDU")
    (risk-tier  "UNACCEPTABLE")
    (article    "Art. 5(1)(f)")
    (reason     "Emotion recognition system deployed in educational institutions (with exceptions for safety purposes)")
    (confidence (min-float ?er ?eu))
  ))
)

(defrule unacceptable-biometric-categorisation
  "Art 5(1)(g): Biometric categorisation inferring sensitive attributes"
  (declare (salience 100))
  (ai-system (biometric-categorisation ?bc&:(> ?bc 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-ART5-G")
    (risk-tier  "UNACCEPTABLE")
    (article    "Art. 5(1)(g)")
    (reason     "Biometric categorisation system inferring race, political opinions, religious beliefs, sexual orientation, or similar sensitive attributes")
    (confidence ?bc)
  ))
)
