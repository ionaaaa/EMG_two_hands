import numpy as np
import pytest

from emg_live_marker.device.protocol import (
    EmgPacket,
    ImuPacket,
    PacketParser,
    build_aa_packet,
    build_bb_packet,
    int16_be_signed,
    int24_be_signed,
)


def test_int24_positive_values():
    assert int24_be_signed(0x00, 0x00, 0x01) == 1
    assert int24_be_signed(0x00, 0x01, 0x00) == 256


def test_int24_negative_values():
    assert int24_be_signed(0xFF, 0xFF, 0xFF) == -1
    assert int24_be_signed(0x80, 0x00, 0x00) == -8_388_608


def test_int16_positive_values():
    assert int16_be_signed(0x00, 0x01) == 1


def test_int16_negative_values():
    assert int16_be_signed(0xFF, 0xFF) == -1
    assert int16_be_signed(0x80, 0x00) == -32_768


def test_aa_packet_decoding():
    parser = PacketParser()
    packets = parser.feed(build_aa_packet(1, [1, 2, 3, 4, 5, 6, 7, 8]))

    assert len(packets) == 1
    packet = packets[0]
    assert isinstance(packet, EmgPacket)
    assert packet.seq == 1
    assert packet.sample_index == 0
    assert packet.t == 0.0
    np.testing.assert_array_equal(packet.values_uv, np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    assert parser.stats.aa_count == 1


def test_aa_packet_negative_decoding():
    parser = PacketParser()
    values = [-1, -100, -1000, 0, 1, 2, 3, 4]
    packets = parser.feed(build_aa_packet(7, values))

    assert len(packets) == 1
    np.testing.assert_array_equal(packets[0].values_uv, np.array(values))


def test_bb_packet_decoding():
    parser = PacketParser()
    packets = parser.feed(build_bb_packet(2, [1000, -1000, 0], [100, -100, 0]))

    assert len(packets) == 1
    packet = packets[0]
    assert isinstance(packet, ImuPacket)
    assert packet.seq == 2
    assert packet.sample_index == 0
    np.testing.assert_allclose(packet.gyro_rad_s, np.array([1.2, -1.2, 0.0]))
    np.testing.assert_allclose(packet.acc_m_s2, np.array([0.05978, -0.05978, 0.0]))
    assert parser.stats.bb_count == 1


def test_sticky_packets_return_two_emg_packets():
    parser = PacketParser()
    data = build_aa_packet(1, [1] * 8) + build_aa_packet(2, [2] * 8)

    packets = parser.feed(data)

    assert len(packets) == 2
    assert [packet.seq for packet in packets] == [1, 2]
    assert [packet.sample_index for packet in packets] == [0, 1]


def test_split_packet_returns_only_after_complete_packet():
    parser = PacketParser()
    packet = build_aa_packet(1, [1] * 8)

    assert parser.feed(packet[:10]) == []
    packets = parser.feed(packet[10:])

    assert len(packets) == 1
    assert packets[0].seq == 1


def test_garbage_bytes_resync_to_valid_packet():
    parser = PacketParser()
    packet = build_aa_packet(1, [3] * 8)

    packets = parser.feed(b"\x00\x11\x22" + packet)

    assert len(packets) == 1
    assert packets[0].seq == 1
    assert parser.stats.bad_header_count == 3
    assert parser.stats.resync_count >= 1


def test_lost_packet_count_uses_global_sequence():
    parser = PacketParser()

    parser.feed(build_aa_packet(1, [1] * 8))
    parser.feed(build_aa_packet(2, [2] * 8))
    parser.feed(build_aa_packet(4, [4] * 8))

    assert parser.stats.global_lost_count == 1
    assert parser.stats.aa_lost_count == 0
    assert parser.stats.bb_lost_count == 0


def test_interleaved_aa_bb_shared_sequence_does_not_count_per_type_loss():
    parser = PacketParser()

    parser.feed(
        build_aa_packet(1, [1] * 8)
        + build_bb_packet(2, [0, 0, 0], [0, 0, 0])
        + build_aa_packet(3, [3] * 8)
        + build_bb_packet(4, [0, 0, 0], [0, 0, 0])
    )

    assert parser.stats.global_lost_count == 0
    assert parser.stats.aa_lost_count == 0
    assert parser.stats.bb_lost_count == 0


def test_global_sequence_loss_counts_skipped_shared_sequence():
    parser = PacketParser()

    parser.feed(
        build_aa_packet(1, [1] * 8)
        + build_bb_packet(2, [0, 0, 0], [0, 0, 0])
        + build_aa_packet(4, [4] * 8)
    )

    assert parser.stats.global_lost_count == 1
    assert parser.stats.aa_lost_count == 0
    assert parser.stats.bb_lost_count == 0


def test_bad_type_resyncs():
    parser = PacketParser()
    bad = bytes([0xD2, 0xD2, 0xD2, 0xCC, 0x01]) + b"\x00" * 24
    valid = build_aa_packet(1, [1] * 8)

    packets = parser.feed(bad + valid)

    assert len(packets) == 1
    assert parser.stats.bad_type_count == 1


def test_builders_return_29_bytes():
    assert len(build_aa_packet(0, [0] * 8)) == 29
    assert len(build_bb_packet(0, [0, 0, 0], [0, 0, 0])) == 29


def test_aa_builder_rejects_wrong_channel_count():
    with pytest.raises(ValueError):
        build_aa_packet(0, [0] * 7)
