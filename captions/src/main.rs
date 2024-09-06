use itertools::Itertools;
use std::{
    collections::{HashMap, HashSet},
    env,
    io::{self, BufRead},
    str::FromStr,
};
use tracing::{debug, error, info, Level};
use tracing_subscriber::fmt::format::FmtSpan;

pub fn log_init() {
    // install global collector configured based on RUST_LOG env var.
    let level =
        env::var("RUST_LOG").map_or(Level::INFO, |v| Level::from_str(&v).unwrap_or(Level::INFO));
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::ACTIVE)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_file(true)
        .with_line_number(true)
        .with_max_level(level)
        .init();
}

fn clean(value: &str) -> String {
    value.chars().filter(|c| c.is_alphanumeric()).collect()
}

fn kmeans(
    file_word_bag: &Vec<(String, HashSet<String>)>,
    known_words: &HashMap<String, f64>,
    num_clusters: usize,
) -> Vec<Vec<(String, HashSet<String>)>> {
    // DenseMatrix wrapper around Vec
    use smartcore::linalg::naive::dense_matrix::DenseMatrix;
    // K-Means
    use smartcore::cluster::kmeans::{KMeans, KMeansParameters};

    // Transform dataset into a NxM matrix
    let values = file_word_bag
        .iter()
        .map(|(_, words)| {
            known_words
                .keys()
                .map(|w| if words.contains(w) { 1.0 } else { 0.0 })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let x = DenseMatrix::from_2d_vec(&values);
    // Fit & predict
    let labels = KMeans::fit(&x, KMeansParameters::default().with_k(num_clusters))
        .and_then(|kmeans| kmeans.predict(&x))
        .unwrap();

    let mut clusters = Vec::new();
    clusters.resize_with(num_clusters, Vec::new);
    for (cluster, (filename, file_words)) in labels.iter().zip(file_word_bag.iter()) {
        clusters
            .get_mut(*cluster as usize)
            .unwrap()
            .push((filename.to_owned(), file_words.to_owned()));
    }
    clusters
}

fn main() -> io::Result<()> {
    log_init();

    let mut known_words = HashMap::<String, usize>::new();
    let mut file_word_bag = Vec::new();
    for filename in env::args_os().skip(1) {
        let file = std::fs::File::open(filename)?;

        let reader = std::io::BufReader::new(file);
        for line in reader.lines().map_while(Result::ok) {
            let (nom, caption) = line.split('\t').skip(1).collect_tuple().unwrap();
            let caption_words = caption
                .split(' ')
                .filter(|c| !c.is_empty())
                .map(clean)
                .collect::<HashSet<_>>();
            for word in caption_words.iter() {
                *known_words.entry(word.clone()).or_default() += 1;
            }
            file_word_bag.push((nom.to_string(), caption_words));
            info!(nom, caption, "file caption");
        }
    }
    let known_words = known_words
        .into_iter()
        .map(|(k, v)| (k, 1.0 - (v as f64 / file_word_bag.len() as f64)))
        .collect::<HashMap<_, _>>();

    let mut idx = 0;
    let mut pending = Vec::new();
    pending.push(file_word_bag);
    while let Some(bag) = pending.pop() {
        if bag.len() <= 100 {
            idx += 1;
            info!(
                idx,
                file_count = bag.len(),
                files = debug(bag),
                "resolved group"
            );
            continue;
        }
        debug!(file_count = bag.len(), "split");
        pending.append(&mut kmeans(&bag, &known_words, 2));
    }
    Ok(())
}
