export const PROFILE_TEMPLATES = [
  'Describe {word} in a sentence.',
  'What does {word} look like?',
  'Explain {word} to a child.',
  'List common traits of {word}.',
  'Where would someone encounter {word}?',
  'What is associated with {word}?',
  'Give an example involving {word}.',
  'Why might someone study {word}?',
  'What makes {word} recognizable?',
  'Describe a typical interaction with {word}.',
  'What questions do people ask about {word}?',
  'Describe the role of {word} in everyday life.',
  'What distinguishes {word} from similar things?',
  'Tell me a fact about {word}.',
  'How would you identify {word}?',
  'Describe a scene involving {word}.',
  'What should a beginner know about {word}?',
  'Name a characteristic feature of {word}.',
  'What words are associated with {word}?',
  'Describe an observation about {word}.',
  'What might surprise someone about {word}?',
  'Explain why {word} is interesting.',
  'How could someone learn about {word}?',
  'Summarize what you know about {word}.',
  'What category does {word} belong to?',
  'Compare {word} with a close alternative.',
  'Which properties of {word} are essential rather than incidental?',
  'What would count as a non-example of {word}?',
  'What is a common mistake when identifying {word}?',
  'How does the context change the meaning of {word}?',
  'What is the origin of the term {word}?',
  'How has human understanding of {word} changed?',
  'What role does {word} play in stories?',
  'How is {word} represented in art?',
  'How might different cultures describe {word}?',
  'What is a historical fact involving {word}?',
  'What sensory details could help someone recognize {word}?',
  'What can be observed directly about {word}?',
  'What evidence would help confirm something is {word}?',
  'How would you explain {word} without a picture?',
  'Write a museum caption about {word}.',
  'Write a dictionary-style entry for {word}.',
  'Write a short encyclopedia introduction to {word}.',
  'Write a classroom question and answer about {word}.',
  'Explain {word} using an analogy, and explain its limitation.',
  'Describe {word} without using the term itself.',
  'What are the parts or components associated with {word}?',
  'What changes can {word} undergo?',
  'What conditions matter for understanding {word}?',
  'What is a cause-and-effect relationship involving {word}?',
  'What can someone predict after encountering {word}?',
  'Describe a realistic problem involving {word}.',
  'What should someone avoid assuming about {word}?',
  'Distinguish a fact from an opinion about {word}.',
  'What is still uncertain or debated about {word}?',
  'What information would an expert seek about {word}?',
  'What is an unusual but accurate example involving {word}?',
  'Give a counterexample to a generalization about {word}.',
  'When might a description of {word} be misleading?',
  'What detail about {word} is often overlooked?',
  'How would you organize a lesson about {word}?',
  'What observation about {word} could be recorded in a notebook?',
  'Explain how {word} relates to its surroundings.',
  'Describe a situation where distinguishing {word} matters.',
]
export const SELECTION_TEMPLATES = [
  'A short definition of {word} is',
  'The most important property of {word} is',
  'An everyday example of {word} is',
]
export const TEST_TEMPLATES = [
  'Someone unfamiliar with {word} should know that',
  'A common misconception about {word} is',
  'To distinguish {word} from a similar concept, the key clue is',
  'A reliable way to explain the meaning of {word} is',
  'An accurate description of {word} must include',
  'If I encountered {word} in a story, I would expect',
  'One observation that provides evidence about {word} is',
  'The relationship between {word} and its broader category is',
  'A concrete situation in which {word} matters is',
  'One incorrect claim about {word}, followed by its correction, is',
  'A useful comparison that clarifies {word} is',
  'An important limitation of a simplified explanation of {word} is',
]
export const renderProbe = (template: string, word: string) =>
  template.replaceAll('{word}', word)

export function parseControls(text: string, concept: string): string[] {
  const cleaned = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim()
  const start = cleaned.indexOf('['),
    end = cleaned.lastIndexOf(']')
  if (start < 0 || end < start)
    throw new Error('The model did not return a JSON array of control words.')
  const value: unknown = JSON.parse(cleaned.slice(start, end + 1))
  const key = (s: string) =>
    s.normalize('NFKC').toLowerCase().trim().replace(/s$/, '')
  if (
    !Array.isArray(value) ||
    value.length !== 3 ||
    value.some(
      (s) =>
        typeof s !== 'string' ||
        !s.trim() ||
        s.trim().length > 40 ||
        /[\n\r\[\]{}]/.test(s),
    )
  )
    throw new Error('Expected exactly three short control words.')
  const words = (value as string[]).map((s) => s.trim())
  const keys = words.map(key)
  if (new Set(keys).size !== 3 || keys.includes(key(concept)))
    throw new Error(
      'Control words must be distinct from each other and the target (including simple plurals).',
    )
  return words
}
