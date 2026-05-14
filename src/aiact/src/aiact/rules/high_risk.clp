;;; EU AI Act - Annex III High-Risk AI Systems
;;; Salience 80

(defrule high-risk-employment
  "Annex III(4): AI used in employment, worker management, and self-employment"
  (declare (salience 80))
  (ai-system (employment-use      ?eu&:(> ?eu 0.4))
             (automates-decisions ?ad&:(> ?ad 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-AIII-4")
    (risk-tier  "HIGH")
    (article    "Annex III, Category 4 | Art. 9-15")
    (reason     "AI system used in employment for recruitment, selection, promotion, task allocation, or performance monitoring that automates decisions affecting individuals")
    (confidence (min-float ?eu ?ad))
  ))
)

(defrule high-risk-employment-indirect
  "Annex III(4): AI used in employment affecting individuals (even without full automation)"
  (declare (salience 80))
  (ai-system (employment-use     ?eu&:(> ?eu 0.4))
             (affects-individuals ?ai&:(> ?ai 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-AIII-4B")
    (risk-tier  "HIGH")
    (article    "Annex III, Category 4 | Art. 9-15")
    (reason     "AI system used in employment contexts that materially affects individuals' access to employment or working conditions")
    (confidence (min-float ?eu ?ai))
  ))
)

(defrule high-risk-biometric-identification
  "Annex III(1): Biometric identification and categorisation systems"
  (declare (salience 80))
  (ai-system (biometric-realtime ?br&:(> ?br 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-AIII-1")
    (risk-tier  "HIGH")
    (article    "Annex III, Category 1 | Art. 9-15")
    (reason     "Biometric identification or categorisation system (post-remote or non-real-time biometrics, e.g. face recognition on recorded footage)")
    (confidence ?br)
  ))
)

(defrule high-risk-critical-infrastructure
  "Annex III(2): AI in critical infrastructure"
  (declare (salience 80))
  (ai-system (critical-infrastructure ?ci&:(> ?ci 0.4))
             (automates-decisions     ?ad&:(> ?ad 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-AIII-2")
    (risk-tier  "HIGH")
    (article    "Annex III, Category 2 | Art. 9-15")
    (reason     "AI system used as safety component of critical infrastructure (roads, water, energy, digital) or as the infrastructure itself")
    (confidence (min-float ?ci ?ad))
  ))
)

(defrule high-risk-education
  "Annex III(3): AI in education affecting access or assessment"
  (declare (salience 80))
  (ai-system (education-use       ?edu&:(> ?edu 0.4))
             (affects-individuals ?ai&:(> ?ai 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-AIII-3")
    (risk-tier  "HIGH")
    (article    "Annex III, Category 3 | Art. 9-15")
    (reason     "AI system determining access to or assigning individuals to educational institutions, or evaluating learning outcomes")
    (confidence (min-float ?edu ?ai))
  ))
)

(defrule high-risk-credit-insurance
  "Annex III(5): AI in access to essential private services"
  (declare (salience 80))
  (ai-system (credit-insurance    ?cr&:(> ?cr 0.4))
             (affects-individuals ?ai&:(> ?ai 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-AIII-5")
    (risk-tier  "HIGH")
    (article    "Annex III, Category 5 | Art. 9-15")
    (reason     "AI system used in creditworthiness evaluation, insurance risk assessment, or decisions about access to essential services")
    (confidence (min-float ?cr ?ai))
  ))
)

(defrule high-risk-law-enforcement
  "Annex III(6): AI used by law enforcement"
  (declare (salience 80))
  (ai-system (law-enforcement ?le&:(> ?le 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-AIII-6")
    (risk-tier  "HIGH")
    (article    "Annex III, Category 6 | Art. 9-15")
    (reason     "AI system used by law enforcement for individual risk assessment, polygraphs, crime analytics, or evidence reliability evaluation")
    (confidence ?le)
  ))
)

(defrule high-risk-migration
  "Annex III(7): AI in migration, asylum, and border control"
  (declare (salience 80))
  (ai-system (migration-border ?mb&:(> ?mb 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-AIII-7")
    (risk-tier  "HIGH")
    (article    "Annex III, Category 7 | Art. 9-15")
    (reason     "AI system used in migration or border control contexts (risk assessment, examination of applications, document verification)")
    (confidence ?mb)
  ))
)

(defrule high-risk-justice
  "Annex III(8): AI in administration of justice and democratic processes"
  (declare (salience 80))
  (ai-system (justice-democratic ?jd&:(> ?jd 0.4)))
  =>
  (assert (matched-rule
    (rule-id    "R-AIII-8")
    (risk-tier  "HIGH")
    (article    "Annex III, Category 8 | Art. 9-15")
    (reason     "AI system used to assist judicial authorities or influence elections and political participation")
    (confidence ?jd)
  ))
)
