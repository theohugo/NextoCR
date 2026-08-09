package org.crforge.bridge;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class BridgeServerTest {

  @Test
  void endpoint_shouldBindLoopbackByDefault() {
    assertThat(BridgeServer.endpoint(9876, false)).isEqualTo("tcp://127.0.0.1:9876");
  }

  @Test
  void endpoint_shouldRequireExplicitRemoteOptIn() {
    assertThat(BridgeServer.endpoint(9876, true)).isEqualTo("tcp://*:9876");
  }
}
