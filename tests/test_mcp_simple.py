"""
Simple MCP Test (Synchronous)

Tests MCP server communication without async complexity.
"""

import os
import sys
import json
import subprocess
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()


def test_github_mcp():
    """Test GitHub MCP server communication."""
    print("=" * 50)
    print("Simple MCP Test (GitHub)")
    print("=" * 50)

    token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        print("[ERROR] GITHUB_PERSONAL_ACCESS_TOKEN not set")
        return False

    print("\n1. Starting GitHub MCP server...")

    env = {**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": token}

    # Start the MCP server
    process = subprocess.Popen(
        ["npx", "-y", "@modelcontextprotocol/server-github"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        shell=True,
    )

    # Collect stderr in background
    stderr_lines = []
    def read_stderr():
        for line in process.stderr:
            stderr_lines.append(line.decode() if isinstance(line, bytes) else line)

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    time.sleep(2)  # Wait for server to start

    if process.poll() is not None:
        print(f"[ERROR] Server exited with code {process.returncode}")
        print(f"Stderr: {''.join(stderr_lines)}")
        return False

    print("   Server started!")

    print("\n2. Sending initialize request...")

    # Send initialize request
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0.0"},
        },
    }

    try:
        message = json.dumps(init_request) + "\n"
        process.stdin.write(message.encode())
        process.stdin.flush()

        # Read response with timeout
        import select

        # On Windows, we need a different approach
        response_line = None
        start_time = time.time()

        while time.time() - start_time < 10:  # 10 second timeout
            # Try to read a line
            try:
                if sys.platform == "win32":
                    # Windows: use threading for timeout
                    result = [None]
                    def read_line():
                        result[0] = process.stdout.readline()

                    read_thread = threading.Thread(target=read_line, daemon=True)
                    read_thread.start()
                    read_thread.join(timeout=1)

                    if result[0]:
                        response_line = result[0]
                        break
                else:
                    response_line = process.stdout.readline()
                    if response_line:
                        break
            except Exception as e:
                print(f"   Read error: {e}")
                break

        if not response_line:
            print("   [TIMEOUT] No response received")
            print(f"   Stderr: {''.join(stderr_lines)}")
            process.kill()
            return False

        response = json.loads(response_line.decode() if isinstance(response_line, bytes) else response_line)
        print(f"   Response: {json.dumps(response, indent=2)[:500]}")

        print("\n3. Sending initialized notification...")

        init_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        process.stdin.write((json.dumps(init_notification) + "\n").encode())
        process.stdin.flush()

        print("\n4. Requesting tools list...")

        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        process.stdin.write((json.dumps(tools_request) + "\n").encode())
        process.stdin.flush()

        # Read tools response
        start_time = time.time()
        while time.time() - start_time < 10:
            result = [None]
            def read_line():
                result[0] = process.stdout.readline()

            read_thread = threading.Thread(target=read_line, daemon=True)
            read_thread.start()
            read_thread.join(timeout=1)

            if result[0]:
                response = json.loads(result[0].decode() if isinstance(result[0], bytes) else result[0])
                if response.get("id") == 2:
                    tools = response.get("result", {}).get("tools", [])
                    print(f"   Found {len(tools)} tools!")
                    for tool in tools[:5]:
                        print(f"   - {tool.get('name')}: {tool.get('description', '')[:50]}...")
                    if len(tools) > 5:
                        print(f"   ... and {len(tools) - 5} more")
                    break

        print("\n5. Shutting down...")
        process.kill()
        process.wait()

        print("\n" + "=" * 50)
        print("[SUCCESS] MCP test passed!")
        print("=" * 50)
        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        process.kill()
        return False


if __name__ == "__main__":
    test_github_mcp()
