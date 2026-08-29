use crate::gpu_actor::GpuActorHandle;

/// Shared across every request. `Clone` is cheap: `GpuActorHandle` is
/// just a channel sender, and `token` is compared, never mutated.
#[derive(Clone)]
pub struct AppState {
    pub gpu: GpuActorHandle,
    /// `None` means this server was started without `--token` — anyone
    /// who can reach it can use it. Fine for same-machine testing or a
    /// private network; the owner's own choice to make, not this
    /// server's to second-guess by refusing to start without one.
    pub token: Option<String>,
}
