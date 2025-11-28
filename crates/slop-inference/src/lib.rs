mod model;
mod pipeline;

pub use model::CLASSIFICATION_THRESHOLD;

use crate::model::MODEL;

/// Run inference on a given text input
/// Returns (labels, ai_probabilities) where ai_probabilities contains P(AI) for each sample
pub fn predict(text: &str) -> ort::Result<Vec<f32>> {
    let session = &MODEL;
    pipeline::predict(session, text)
}

/// Test function with hardcoded text for testing purposes
/// Returns (labels, ai_probabilities) where ai_probabilities contains P(AI) for each sample
pub fn infer() -> ort::Result<Vec<f32>> {
    let input = "Advice about receiving advice\n\nDoing something one way and then realizing there was a great amount of other options is a very annoying (and unfortunately, very common) case. It's even more annoying when you don't realize what the other solution was because then you can never grow as a person. People should ask more than one person for advice when seeking it because not everything will work for everyone and it's always better to hear from different sources.\n\nNot everything will work for everyone, as everybody has their own way of dealing with certain things. Say there's a student in algebra honors and they are studying for a test that is stressing them out very much. Now, this student has trouble paying attention in class, and as a result of that, they often don't have the best of notes. If the student's parent is giving them advice and the parent says to the student that they should check their notes, the student will most likely have to seek help elsewhere because that advice is not very good for them in particular.\n\nIt's always better to hear from different sources. There are lots of times when you need to seek multiple solutions. In science labs, multiple trials are required to find out if the initial hypothesis was correct. When writing about current events, one must gather information from different sources or they could be risking the possibility of being biased. It's no different when people are seeking advice from someone. If somebody receives advice from one person and they aren't happy with outcome of the advice they took, chances are they didn't hear from enough people and a different solution was just waiting for them to try it out.\n\nAt the end of the day, receiving advice from multiple people is the best thing to do because not everything will work for everyone and it is always better to hear from more than one source.";
    let input = "So I have a working implementation of my slop detection tool. It works end to end and now what I have done is made a pipeline to build the training dataset (before I was using a single kaggle dataset but I have now build notebooks/dataset_curation.ipynb) a wide variety of data sources from huggingface. As you will see these datasources vary but the key being we have a bunch of different data sources (some a mix of ai and human text, some just human and some just ai text). What I want to do now is really try and optimise my training pipeline to produce a really good model (or even set of models).



What I really want you to do first is understand the problem I am trying to solve and how I am currently going about it and why (why not) it makes sense to go the route I am going.



The whole idea of this kinda stems from the fact that I saw a kaggle competition from a few years ago and most of the best solutions created there own tokeniser (which makes a lot of sense for this type of problem. This got me thinking why dont we just use and existing tokeniser (like the open ai one tiktoken) as our tokeniser. I think my intuition for this is that llms talk via this kinda filter layer (the tokeniser, inputs and outputs get tokenised) This kinda gets me thinking that the way llms speak would leave some sort of signal that we should be able to detect in this tokeniser layer (because it feels intuitively that the llm speak would pass easier through this layer (the shapes created (tokens from words and sentences) much easier than the messy/ inconsistent, missspelt output humans produce. My thinking therefore was if we use these pre-existing tokenisers (and specifically Open-AIs tiktoken as it has the largest market share (most slop will probably be some form of open ai model as they are the largest ai company)) as our kinda input, we might be able to detect artefacts at this layer which would help us classify between ai and human text.



A few things we do are generate token ngrams. At the moment they are set to (3, 5) but maybe (2,4) would be better as longer chains of unusual tokens are less likely. What I really want to try and hone down on and really try and exploit is that the AI's will likely use a similar phrases or sentence structures or even words (made up of 1-many tokens) that will often be usual or 'clean' tokens (more used patterns). Where as I can imagine humans get a bit more messy.";

    predict(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer() {
        let probs = infer().unwrap();
        let labels: Vec<i64> = probs
            .iter()
            .map(|&p| i64::from(p >= CLASSIFICATION_THRESHOLD as f32))
            .collect();

        println!("Labels: {labels:?}");
        println!("Probabilities: {probs:?}");
    }
}
