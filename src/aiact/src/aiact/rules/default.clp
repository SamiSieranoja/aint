;;; EU AI Act - Fallback rule: Minimal / No Risk
;;; Fires only when no higher-tier rule has asserted a matched-rule fact

(defrule default-minimal-risk
  "No prohibited, high-risk, or transparency indicators found"
  (declare (salience -100))
  (ai-system)
  (not (matched-rule (risk-tier "UNACCEPTABLE")))
  (not (matched-rule (risk-tier "HIGH")))
  (not (matched-rule (risk-tier "LIMITED")))
  =>
  (assert (matched-rule
    (rule-id    "R-DEFAULT")
    (risk-tier  "MINIMAL")
    (article    "N/A")
    (reason     "No high-risk, prohibited, or transparency-obligation indicators found. System falls in the minimal / no risk category with no specific EU AI Act obligations (though general product safety and GDPR may still apply).")
    (confidence 0.9)
  ))
)
