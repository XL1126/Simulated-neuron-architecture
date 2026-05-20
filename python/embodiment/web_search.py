import urllib.request
import urllib.parse
import json

class WebSearchAPI:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.user_agent = "SNA/1.0 (Simulated Neuron Architecture)"

    def query(self, search_query, max_results=5):
        if not self.enabled:
            return "[搜索不可用]"

        try:
            encoded_query = urllib.parse.quote_plus(search_query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8', errors='ignore')

            results = self._parse_duckduckgo_html(html)
            if not results:
                return f"未找到关于 '{search_query}' 的结果。"

            output = f"搜索结果 ({len(results[:max_results])}):\n"
            for i, (title, snippet, link) in enumerate(results[:max_results], 1):
                output += f"\n[{i}] {title}\n    {snippet}\n"

            return output

        except Exception as e:
            return f"[搜索错误]: {e}"

    def _parse_duckduckgo_html(self, html):
        results = []
        try:
            from html.parser import HTMLParser

            class DDGParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.results = []
                    self.in_result = False
                    self.in_title = False
                    self.in_snippet = False
                    self.current_title = ""
                    self.current_snippet = ""
                    self.current_link = ""

                def handle_starttag(self, tag, attrs):
                    attrs_dict = dict(attrs)
                    if tag == 'a' and 'result__a' in attrs_dict.get('class', ''):
                        self.in_result = True
                        self.in_title = True
                        self.current_link = attrs_dict.get('href', '')
                    elif tag == 'a' and 'result__snippet' in attrs_dict.get('class', ''):
                        self.in_snippet = True

                def handle_data(self, data):
                    if self.in_title:
                        self.current_title += data
                    if self.in_snippet:
                        self.current_snippet += data

                def handle_endtag(self, tag):
                    if tag == 'a' and self.in_title:
                        self.in_title = False
                    if tag == 'a' and self.in_snippet:
                        self.in_snippet = False
                        self.results.append((
                            self.current_title.strip(),
                            self.current_snippet.strip(),
                            self.current_link
                        ))
                        self.current_title = ""
                        self.current_snippet = ""
                        self.current_link = ""

            parser = DDGParser()
            parser.feed(html)
            results = parser.results
        except Exception:
            pass

        return results
