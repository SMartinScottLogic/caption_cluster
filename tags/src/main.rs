use std::{
    collections::{HashMap, HashSet},
    env,
    fs::File,
    io::{self, BufRead as _, BufReader},
    path::PathBuf,
    str::FromStr,
};

use itertools::Itertools;
use kmedoids::ArrayAdapter;
use tracing::{debug, error, info, instrument, Level};
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

#[derive(Debug)]
struct TaggedFile {
    filename: String,
    tags: HashMap<String, f64>,
}
fn load_tags(filename: &str) -> io::Result<Vec<TaggedFile>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let tagged_files = reader
        .lines()
        .flatten()
        .map(|line| {
            let (_, image, tags, ratings) = line.split('\t').collect_tuple().unwrap();
            let mut tags = tags
                .split(',')
                .filter(|t| !t.is_empty())
                .flat_map(Tag::from_str)
                .fold(HashMap::new(), |mut acc, tag| {
                    acc.insert(tag.name, tag.score);
                    acc
                });
            // Normalize
            let len = tags.values().map(|v| v * v).sum::<f64>().sqrt();
            for (_, v) in tags.iter_mut() {
                *v /= len;
            }
            debug!(image, tags = debug(&tags), ratings);
            TaggedFile {
                filename: image.to_string(),
                tags,
            }
        })
        .filter(|t| !t.tags.is_empty())
        .collect_vec();

    Ok(tagged_files)
}

#[derive(Debug)]
struct Tag {
    name: String,
    score: f64,
}
impl FromStr for Tag {
    type Err = std::convert::Infallible;

    fn from_str(t: &str) -> std::result::Result<Self, <Self as std::str::FromStr>::Err> {
        info!(t, "to_tags");
        let t = t.trim();
        let r = regex::Regex::new(r"^\((?P<name>.+):(?P<score>-?[0-9\.]+)\)$").unwrap();
        let c = r.captures(t).unwrap();
        let name = c.name("name").unwrap().as_str().to_string();
        let score = c.name("score").unwrap().as_str().parse().unwrap();
        let tag = Self { name, score };
        Ok(tag)
    }
}

struct Dissim<'a> {
    tagged_files: &'a [TaggedFile],
}
impl<'a> Dissim<'a> {
    fn new(tagged_files: &'a [TaggedFile]) -> Self {
        Self { tagged_files }
    }
}
impl<'a> ArrayAdapter<f64> for Dissim<'a> {
    fn len(&self) -> usize {
        self.tagged_files.len()
    }

    fn is_square(&self) -> bool {
        true
    }

    fn get(&self, x: usize, y: usize) -> f64 {
        let tags_x = &self.tagged_files[x].tags;
        debug!(x, tags_x = debug(tags_x));

        let tags_y = &self.tagged_files[y].tags;
        debug!(y, tags_y = debug(tags_y));

        let similarity: f64 = tags_x
            .keys()
            .chain(tags_y.keys())
            .cloned()
            .unique()
            .map(|key| tags_x.get(&key).unwrap_or(&0.) * tags_y.get(&key).unwrap_or(&0.))
            .sum();

        debug!(x, y, similarity);

        if x == y && (1.0 - similarity).abs() > 1e-3 {
            error!(
                x,
                tags_x = debug(tags_x),
                y,
                tags_y = debug(tags_y),
                similarity
            );
            panic!();
        }

        1. - similarity
    }
}

#[instrument(skip(tagged_files))]
fn cut(tagged_files: &[TaggedFile]) -> HashMap<usize, (HashSet<String>, HashSet<String>)> {
    // Aim for ~50 files per cluster
    let k = tagged_files.len() / 50;
    let mat = Dissim::new(tagged_files);
    let mut meds = kmedoids::random_initialization(tagged_files.len(), k, &mut rand::thread_rng());
    let r: (f64, Vec<usize>, usize, usize) = kmedoids::fasterpam(&mat, &mut meds, k);

    info!(r = debug(&r), "medoids");

    let o =
        r.1.iter()
            .enumerate()
            .fold(HashMap::new(), |mut acc, (id, v)| {
                let t = tagged_files.get(id).unwrap();
                let entry = acc
                    .entry(*v)
                    .or_insert_with(|| (HashSet::new(), HashSet::new()));
                entry.1.extend(t.tags.iter().map(|t| t.0.to_owned()));
                entry.0.insert(t.filename.clone());
                acc
            });

    o
}

fn main() -> io::Result<()> {
    log_init();

    let tagged_files = std::env::args()
        .skip(1)
        .flat_map(|f| load_tags(&f))
        .flatten()
        .collect_vec();

    let mut tag_count: HashMap<String, usize> = HashMap::new();
    for tagged_file in &tagged_files {
        for tag in tagged_file.tags.keys() {
            *tag_count.entry(tag.to_string()).or_default() += 1;
        }
    }
    info!(tag_count = debug(tag_count), "counts");
    //info!(tagged_files = debug(&tagged_files), "tagged files");
    let partition = cut(&tagged_files);

    let write_files = true;
    let move_files = true;
    for (k, (files, v)) in partition {
        let outdir = PathBuf::from(format!("tag_partitioned/group_{k}"));

        for file in &files {
            let to = outdir.join(PathBuf::from(file).file_name().unwrap());
            debug!(k, files = debug(&files), tags = debug(&v));

            info!("copy {file} -> {to:?}");
            if write_files {
                std::fs::create_dir_all(&outdir)?;

                // Assume error means no such file
                if std::fs::metadata(&to).is_err() {
                    if std::fs::copy(file, to).is_ok() {
                        if move_files {
                            info!("remove {file}");
                            std::fs::remove_file(file)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
