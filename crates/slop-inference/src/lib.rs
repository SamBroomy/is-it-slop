mod model;
mod pipeline;

use crate::model::MODEL;

/// Run inference on a given text input
pub fn predict(text: &str) -> ort::Result<(Vec<i64>, Vec<Vec<f32>>)> {
    let session = &MODEL;
    pipeline::predict(session, text)
}

/// Test function with hardcoded text for testing purposes
pub fn infer() -> ort::Result<(Vec<i64>, Vec<Vec<f32>>)> {
    let input = "Advice about receiving advice\n\nDoing something one way and then realizing there was a great amount of other options is a very annoying (and unfortunately, very common) case. It's even more annoying when you don't realize what the other solution was because then you can never grow as a person. People should ask more than one person for advice when seeking it because not everything will work for everyone and it's always better to hear from different sources.\n\nNot everything will work for everyone, as everybody has their own way of dealing with certain things. Say there's a student in algebra honors and they are studying for a test that is stressing them out very much. Now, this student has trouble paying attention in class, and as a result of that, they often don't have the best of notes. If the student's parent is giving them advice and the parent says to the student that they should check their notes, the student will most likely have to seek help elsewhere because that advice is not very good for them in particular.\n\nIt's always better to hear from different sources. There are lots of times when you need to seek multiple solutions. In science labs, multiple trials are required to find out if the initial hypothesis was correct. When writing about current events, one must gather information from different sources or they could be risking the possibility of being biased. It's no different when people are seeking advice from someone. If somebody receives advice from one person and they aren't happy with outcome of the advice they took, chances are they didn't hear from enough people and a different solution was just waiting for them to try it out.\n\nAt the end of the day, receiving advice from multiple people is the best thing to do because not everything will work for everyone and it is always better to hear from more than one source.";

    predict(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer() {
        let (labels, probs) = infer().unwrap();
        println!("Labels: {labels:?}");
        println!("Probabilities: {probs:?}");
    }
}
