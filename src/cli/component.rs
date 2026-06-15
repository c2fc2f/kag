//! Validated component identifiers shared across the subcommands
//!
//! Provides [`ComponentName`], a newtype over `String` that enforces a strict
//! identifier format (non-empty, lowercase ASCII alphanumerics and hyphens)
//! whenever a value is parsed via [`FromStr`] or deserialized. The same check
//! therefore guards both CLI input and configuration files

use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize};

/// A validated, strictly formatted component identifier.
///
/// This type wraps a `String` and guarantees that the identifier complies
/// with system constraints upon deserialization. It implements [`Deref`] to
/// allow seamless usage as a standard string slice (`&str`)
///
/// # Validation Rules
/// - Cannot be empty
/// - No spaces allowed
/// - No special characters allowed (except hyphens)
/// - Strictly lowercase alphanumeric characters (`a-z`, `0-9`, `-`)
#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct ComponentName(String);

impl FromStr for ComponentName {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    if s.is_empty() {
      return Err("Component name cannot be empty".to_string());
    }

    if !s
      .chars()
      .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
      return Err(
        "\
          Component name must contain only lowercase alphanumeric \
          characters and hyphens\
        "
        .to_string(),
      );
    }

    Ok(Self(s.to_string()))
  }
}

impl<'de> Deserialize<'de> for ComponentName {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    Self::from_str(&String::deserialize(deserializer)?)
      .map_err(serde::de::Error::custom)
  }
}

impl Serialize for ComponentName {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    serializer.serialize_str(&self.0)
  }
}

impl std::ops::Deref for ComponentName {
  type Target = str;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl std::fmt::Debug for ComponentName {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Debug::fmt(&self.0, f)
  }
}

impl std::fmt::Display for ComponentName {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    std::fmt::Display::fmt(&self.0, f)
  }
}
