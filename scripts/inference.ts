import { HfInference } from "@huggingface/inference";
import { HF_ACCESS_TOKEN } from "../constants";

const hf = new HfInference(HF_ACCESS_TOKEN);

/**
 * A range of model interactions as per example docs in the HuggingFace Inference API
 * @see https://huggingface.co/docs/huggingface.js/inference/README
 */
const main = async () => {
  const response = await hf.fillMask({
    model: "bert-base-uncased",
    inputs: "[MASK] world!",
  });

  //   console.log(response);

  const response2 = await hf.summarization({
    model: "facebook/bart-large-cnn",
    inputs:
      "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930.",
    parameters: {
      min_length: 10,
      max_length: 30,
    },
  });

  //   console.log(response2);

  const response3 = await hf.textClassification({
    model: "distilbert-base-uncased-finetuned-sst-2-english",
    inputs: "I like you. I love you.",
  });

  //   console.log(response3);

  const response4 = await hf.textGeneration({
    model: "gpt2",
    inputs: "The number of letters in the alphabet is",
  });

  //   console.log(response4);

  const response5 = await hf.tokenClassification({
    model: "dbmdz/bert-large-cased-finetuned-conll03-english",
    inputs: "My name is Sarah Jessica Parker but you can call me Jessica",
  });

  //   console.log(response5);

  const response6 = await hf.translation({
    model: "t5-base",
    inputs: "My name is Wolfgang and I live in Berlin",
  });

  //   console.log(response6);

  const response7 = await hf.zeroShotClassification({
    model: "facebook/bart-large-mnli",
    inputs: [
      "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!",
    ],
    parameters: { candidate_labels: ["refund", "legal", "faq"] },
  });

  // console.log(response7);

  const response8 = await hf.conversational({
    model: "microsoft/DialoGPT-large",
    inputs: {
      past_user_inputs: ["Which movie is the best ?"],
      generated_responses: ["It is Die Hard for sure."],
      text: "Can you explain why ?",
    },
  });

  // console.log(response8);

  const response9 = await hf.sentenceSimilarity({
    model: "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
    inputs: {
      source_sentence: "That is a happy person",
      sentences: [
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day",
      ],
    },
  });

  // console.log(response9);

  console.log("Done");
  process.exit(0);
};

main();
