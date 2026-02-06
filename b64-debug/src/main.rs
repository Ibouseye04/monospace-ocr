use base64::{Engine as _, engine::general_purpose::STANDARD};
use clap::Parser;
use miette::{Diagnostic, NamedSource, Result, SourceSpan};
use std::io::{self, BufRead, BufReader, Write};
use thiserror::Error;

#[derive(Parser)]
#[command(author, version, about = "base64 validator")]
struct CliOpts {
    /// Emulate `base64 -d` behavior (fail on spaces/tabs, require length multiple of 4)
    #[arg(short, long)]
    strict: bool,
    /// Omit decoded output, akin to redirecting to /dev/null
    #[arg(short, long)]
    quiet: bool,
}

#[derive(Error, Debug, Diagnostic)]
#[error("Base64 decoding failed")]
#[diagnostic(code(base64::decode_error))]
enum Base64Error {
    #[error("Invalid character '{received}' (hex: 0x{received_hex:02x})")]
    #[help("In strict mode, only newlines are ignored. Spaces and tabs are prohibited.")]
    InvalidByte {
        received: char,
        received_hex: u8,
        #[label("this character is not allowed here")]
        span: SourceSpan,
    },

    #[error("Invalid length: input must be a multiple of 4")]
    #[help("Standard Base64 requires padding ('=') to reach a length multiple of 4.")]
    InvalidLength {
        #[label("this string has {len} non-whitespace chars; expected a multiple of 4")]
        span: SourceSpan,
        len: usize,
    },

    #[error("Invalid last symbol: bits remaining in '{received}'")]
    #[help(
        "This character has trailing bits that don't form a full byte. Data is likely corrupted."
    )]
    InvalidLastSymbol {
        received: char,
        #[label("trailing bits here")]
        span: SourceSpan,
    },

    #[error("Invalid padding")]
    #[help("Padding ('=') must only appear at the very end of the stream.")]
    InvalidPadding {
        #[label("misplaced padding")]
        span: SourceSpan,
    },
}

fn process_base64_stream<R: BufRead>(mut reader: R, strict: bool, quiet: bool) -> Result<()> {
    let mut raw_input = String::new();
    let mut clean_buffer = Vec::new();
    let mut offset_map = Vec::new();

    let mut line = String::new();
    let mut current_pos = 0;

    while reader
        .read_line(&mut line)
        .map_err(|e| miette::miette!(e))?
        > 0
    {
        raw_input.push_str(&line);
        for (i, ch) in line.char_indices() {
            let is_newline = ch == '\n' || ch == '\r';
            let is_other_whitespace = ch.is_whitespace() && !is_newline;

            if is_newline {
                continue; // Always ignore newlines
            }

            if strict && is_other_whitespace {
                // In strict mode, whitespace (aside from line breaks) triggers an error
                let report = Base64Error::InvalidByte {
                    received: ch,
                    received_hex: ch as u8,
                    span: (current_pos + i, 1).into(),
                };
                return Err(miette::Report::new(report)
                    .with_source_code(NamedSource::new("stdin", raw_input)));
            }

            if !ch.is_whitespace() {
                clean_buffer.push(ch as u8);
                offset_map.push(current_pos + i);
            }
        }
        current_pos += line.len();
        line.clear();
    }

    if clean_buffer.is_empty() {
        return Ok(());
    }

    // In strict mode, validate input length (i.e. require padding)
    if strict && clean_buffer.len() % 4 != 0 {
        let last_phys_idx = offset_map.last().copied().unwrap_or(0);
        let report = Base64Error::InvalidLength {
            len: clean_buffer.len(),
            span: (0, last_phys_idx + 1).into(),
        };
        return Err(
            miette::Report::new(report).with_source_code(NamedSource::new("stdin", raw_input))
        );
    }

    match STANDARD.decode(&clean_buffer) {
        Ok(decoded) => {
            eprintln!(
                "Valid Base64. {} chars decoded to {} bytes",
                clean_buffer.len(),
                decoded.len()
            );
            if !quiet {
                let _ = std::io::stdout().lock().write_all(&decoded);
            }
        }
        Err(e) => {
            let report = match e {
                base64::DecodeError::InvalidByte(idx, byte) => Base64Error::InvalidByte {
                    received: byte as char,
                    received_hex: byte,
                    span: (offset_map[idx], 1).into(),
                },
                base64::DecodeError::InvalidLength(idx) => Base64Error::InvalidLength {
                    len: clean_buffer.len(),
                    span: (offset_map[idx.saturating_sub(1)], 1).into(),
                },
                base64::DecodeError::InvalidLastSymbol(idx, byte) => {
                    Base64Error::InvalidLastSymbol {
                        received: byte as char,
                        span: (offset_map[idx], 1).into(),
                    }
                }
                base64::DecodeError::InvalidPadding => {
                    let first_pad = clean_buffer.iter().position(|&b| b == b'=').unwrap_or(0);
                    Base64Error::InvalidPadding {
                        span: (offset_map[first_pad], 1).into(),
                    }
                }
            };
            return Err(
                miette::Report::new(report).with_source_code(NamedSource::new("stdin", raw_input))
            );
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let opts = CliOpts::parse();
    let stdin = io::stdin();
    let reader = BufReader::new(stdin.lock());

    process_base64_stream(reader, opts.strict, opts.quiet)?;

    Ok(())
}
