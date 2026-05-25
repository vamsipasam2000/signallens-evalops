# SignalLens Trust-Safety Policy

This Day 1 policy is intentionally small. It provides static grounding for the API and prepares the project for a LangGraph workflow in Day 2.

## safe

Definition: Content that does not contain abuse, spam, or urgent safety risk.

Example: "This product was easy to set up and the docs helped."

Recommended action: `allow`

## spam

Definition: Repetitive, promotional, deceptive, or low-value content that may reduce feed quality.

Example: "Click this link now for guaranteed free followers."

Recommended action: `downrank`

## harassment

Definition: Abusive, targeted, or threatening content directed at a person or group.

Example: "You are worthless and everyone should attack your account."

Recommended action: `block`

## self_harm_sensitive

Definition: Content suggesting potential self-harm, crisis, or acute distress.

Example: "I do not want to be alive anymore."

Recommended action: `human_review`

