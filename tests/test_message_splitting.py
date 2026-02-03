import pytest
from qqbot.streaming import split_message_stream


class TestSplitMessageStream:
    """Test suite for split_message_stream function."""
    
    def test_empty_input(self):
        """Test with empty buffer and chunk."""
        ready, buffer = split_message_stream("", "")
        assert ready == []
        assert buffer == ""
    
    def test_no_separator(self):
        """Test when there's no \\n\\n separator."""
        ready, buffer = split_message_stream("", "Hello world")
        assert ready == []
        assert buffer == "Hello world"
    
    def test_accumulate_multiple_chunks_no_separator(self):
        """Test accumulating multiple chunks without separator."""
        buffer = ""
        buffer = split_message_stream(buffer, "Hello ")[1]
        buffer = split_message_stream(buffer, "world ")[1]
        ready, buffer = split_message_stream(buffer, "test")
        
        assert ready == []
        assert buffer == "Hello world test"
    
    def test_single_part_under_threshold(self):
        """Test single part under 400 characters with separator."""
        text = "Short message"  # < 400 chars
        ready, buffer = split_message_stream("", text + "\n\n")
        
        assert ready == []
        assert buffer == text
    
    def test_single_part_over_threshold(self):
        """Test single part over 400 characters."""
        text = "a" * 450  # > 400 chars
        ready, buffer = split_message_stream("", text + "\n\n")
        
        assert len(ready) == 1
        assert ready[0] == text
        assert buffer == ""
    
    def test_multiple_parts_all_over_threshold(self):
        """Test multiple parts all exceeding threshold."""
        text1 = "a" * 450
        text2 = "b" * 500
        text3 = "c" * 420
        
        input_text = f"{text1}\n\n{text2}\n\n{text3}\n\n"
        ready, buffer = split_message_stream("", input_text)
        
        assert len(ready) == 3
        assert ready[0] == text1
        assert ready[1] == text2
        assert ready[2] == text3
        assert buffer == ""
    
    def test_accumulate_small_parts(self):
        """Test accumulating small parts until threshold is reached."""
        part1 = "a" * 100  # < 400
        part2 = "b" * 100  # < 400
        part3 = "c" * 250  # combined with part1+part2 would be > 400
        
        input_text = f"{part1}\n\n{part2}\n\n{part3}\n\n"
        ready, buffer = split_message_stream("", input_text)
        
        # All parts combine together since they're all processed in one call
        # part1+part2 = 200 < 400, then + part3 = 450 > 400, so emits all
        assert len(ready) == 1
        assert ready[0] == f"{part1}\n\n{part2}\n\n{part3}"
        assert buffer == ""
    
    def test_streaming_scenario(self):
        """Test realistic streaming scenario."""
        buffer = ""
        all_ready = []
        
        # First chunk - short
        ready, buffer = split_message_stream(buffer, "Hello ")
        all_ready.extend(ready)
        assert len(all_ready) == 0
        assert buffer == "Hello "
        
        # Second chunk - still accumulating
        ready, buffer = split_message_stream(buffer, "world\n\n")
        all_ready.extend(ready)
        assert len(all_ready) == 0
        assert buffer == "Hello world"
        
        # Third chunk - add more text
        ready, buffer = split_message_stream(buffer, "This is ")
        all_ready.extend(ready)
        assert len(all_ready) == 0
        
        # Fourth chunk - add long text that exceeds threshold
        long_text = "x" * 400
        ready, buffer = split_message_stream(buffer, long_text + "\n\n")
        all_ready.extend(ready)
        assert len(all_ready) == 1
        assert "Hello world" in all_ready[0]
        assert buffer == ""
    
    def test_custom_min_length(self):
        """Test with custom minimum length."""
        text = "a" * 150  # < 400 but > 100
        ready, buffer = split_message_stream("", text + "\n\n", min_length=100)
        
        assert len(ready) == 1
        assert ready[0] == text
        assert buffer == ""
    
    def test_empty_parts_skipped(self):
        """Test that empty parts are skipped."""
        text1 = "a" * 450
        text2 = "b" * 450
        
        # Multiple consecutive separators create empty parts
        input_text = f"{text1}\n\n\n\n{text2}\n\n"
        ready, buffer = split_message_stream("", input_text)
        
        assert len(ready) == 2
        assert ready[0] == text1
        assert ready[1] == text2
    
    def test_incomplete_last_part(self):
        """Test handling of incomplete last part."""
        text1 = "a" * 450
        text2 = "incomplete"
        
        input_text = f"{text1}\n\n{text2}"  # No trailing separator
        ready, buffer = split_message_stream("", input_text)
        
        assert len(ready) == 1
        assert ready[0] == text1
        assert buffer == text2
    
    def test_progressive_accumulation(self):
        """Test progressive accumulation with multiple calls."""
        buffer = ""
        all_ready = []
        
        # Add 5 small chunks
        for i in range(5):
            text = f"Part {i} " + ("x" * 80)
            ready, buffer = split_message_stream(buffer, text + "\n\n")
            all_ready.extend(ready)
        
        # Should accumulate until threshold
        assert len(all_ready) >= 1
        assert len(buffer) > 0 or len(all_ready) > 0
    
    def test_exactly_at_threshold(self):
        """Test with text exactly at threshold."""
        text = "a" * 400  # Exactly 400
        ready, buffer = split_message_stream("", text + "\n\n")
        
        assert len(ready) == 1
        assert ready[0] == text
        assert buffer == ""
    
    def test_just_below_threshold(self):
        """Test with text just below threshold."""
        text = "a" * 399  # Just below 400
        ready, buffer = split_message_stream("", text + "\n\n")
        
        assert ready == []
        assert buffer == text
    
    def test_multiple_separators_in_chunk(self):
        """Test chunk containing multiple separators."""
        text1 = "a" * 450
        text2 = "b" * 450
        text3 = "c" * 450
        
        # All in one chunk
        chunk = f"{text1}\n\n{text2}\n\n{text3}\n\n"
        ready, buffer = split_message_stream("", chunk)
        
        assert len(ready) == 3
        assert buffer == ""
    
    def test_buffer_persists_across_calls(self):
        """Test that buffer correctly persists across multiple calls."""
        buffer = ""
        
        # First call - accumulate
        ready, buffer = split_message_stream(buffer, "Small\n\n")
        assert ready == []
        assert buffer == "Small"
        
        # Second call - still accumulating (note: separators are consumed during split)
        ready, buffer = split_message_stream(buffer, "Another\n\n")
        assert ready == []
        assert buffer == "SmallAnother"
        
        # Third call - push over threshold
        big_text = "x" * 400
        ready, buffer = split_message_stream(buffer, big_text + "\n\n")
        assert len(ready) == 1
        assert "Small" in ready[0]
        assert buffer == ""
    
    def test_real_world_scenario(self):
        """Test a real-world streaming scenario with varied content."""
        buffer = ""
        all_messages = []
        
        chunks = [
            "你好！",
            "我是一个AI助手。\n\n",
            "让我为你讲一个",
            "很长很长的故事。" + ("这是故事的内容。" * 50) + "\n\n",  # Increase to ensure > 400 chars
            "故事讲完了。",
            "\n\n谢谢阅读！"
        ]
        
        for chunk in chunks:
            ready, buffer = split_message_stream(buffer, chunk)
            all_messages.extend(ready)
        
        # Final flush
        if buffer:
            all_messages.append(buffer)
        
        # Should produce at least 1 message, and at least one should be >= 400 chars
        assert len(all_messages) >= 1
        assert any(len(msg) >= 400 for msg in all_messages)
    
    def test_final_message_under_threshold(self):
        """Test that final message under 400 chars is still returned in buffer."""
        buffer = ""
        
        # Send a long message first
        long_text = "x" * 450
        ready, buffer = split_message_stream(buffer, long_text + "\n\n")
        assert len(ready) == 1
        assert buffer == ""
        
        # Then send a short final message
        short_final = "This is the end."  # < 400 chars
        ready, buffer = split_message_stream(buffer, short_final)
        
        # Should not be emitted yet (no separator)
        assert ready == []
        assert buffer == short_final
        
        # This buffer would be sent in the final flush in actual usage
        # Simulating the final flush:
        if buffer:
            final_message = buffer
            assert final_message == short_final
            assert len(final_message) < 400  # Confirms it's under threshold but still sent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
