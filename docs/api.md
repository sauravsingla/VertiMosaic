# API Reference

This reference exposes the primary research classes and protocol surfaces. The command-line interfaces remain the recommended path for reproducible experiments because they capture configuration and evidence more consistently than ad-hoc interactive calls.

## Models

::: vertimosaic.models.VFLLogisticRegression
    options:
      show_root_heading: true
      members: true

::: vertimosaic.models.VFLHistGBDT
    options:
      show_root_heading: true
      members: true

## Parties

::: vertimosaic.parties.ActiveParty
    options:
      show_root_heading: true
      members: true

::: vertimosaic.parties.PassiveParty
    options:
      show_root_heading: true
      members: true

::: vertimosaic.parties.RemotePassiveParty
    options:
      show_root_heading: true
      members: true

::: vertimosaic.parties.RemotePartyService
    options:
      show_root_heading: true
      members: true

## Transport

::: vertimosaic.transport.InMemoryTransport
    options:
      show_root_heading: true
      members: true

::: vertimosaic.transport.RemoteHTTPTransport
    options:
      show_root_heading: true
      members: true

## Privacy research backends

::: vertimosaic.privacy.GaussianDPBackend
    options:
      show_root_heading: true
      members: true

::: vertimosaic.privacy.ClippedGaussianDPBackend
    options:
      show_root_heading: true
      members: true

::: vertimosaic.privacy.PairwiseMaskSecureAggregation
    options:
      show_root_heading: true
      members: true

::: vertimosaic.privacy.AdditiveSecretSharingSum
    options:
      show_root_heading: true
      members: true

::: vertimosaic.privacy.OpenMinedPSIBackend
    options:
      show_root_heading: true
      members: true

::: vertimosaic.privacy.PaillierHomomorphicSum
    options:
      show_root_heading: true
      members: true

!!! warning "Scope the guarantee"
    The presence of a privacy primitive in the API does not imply that unrelated protocol messages inherit its guarantee. Use [Privacy Boundaries](privacy_boundaries.md) and [Threat Model](threat_model.md) when describing an experiment.
