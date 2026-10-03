// Static mock-exam bank (master spec 11.1): the answers are stored with the session, never graded here.
export const EXAM_MINUTES = 15;
export const QUESTIONS: { id: string; text: string; options: string[] }[] = [
  { id: "q1", text: "Which number comes next: 2, 6, 12, 20, 30, ...?", options: ["36", "40", "42", "44"] },
  { id: "q2", text: "A train covers 180 km in 2 h 15 min. Its average speed is:", options: ["72 km/h", "80 km/h", "85 km/h", "90 km/h"] },
  { id: "q3", text: "Which word does not belong?", options: ["Cedar", "Maple", "Tulip", "Birch"] },
  { id: "q4", text: "If all blips are blops and some blops are blups, which must be true?", options: ["All blups are blips", "Some blips are blups", "All blips are blops", "No blops are blips"] },
  { id: "q5", text: "What is 15% of 240?", options: ["24", "32", "36", "40"] },
  { id: "q6", text: "A clock shows 3:15. The angle between its hands is:", options: ["0 deg", "7.5 deg", "15 deg", "22.5 deg"] },
  { id: "q7", text: "Which is the odd one out?", options: ["Triangle", "Square", "Cube", "Pentagon"] },
  { id: "q8", text: "Rearrange 'TEAPLN' to get a:", options: ["Planet", "Animal", "Country", "Fruit"] },
  { id: "q9", text: "The probability of two heads in two fair coin tosses is:", options: ["1/2", "1/3", "1/4", "1/8"] },
  { id: "q10", text: "Which binary number equals decimal 13?", options: ["1011", "1101", "1110", "1001"] },
];
