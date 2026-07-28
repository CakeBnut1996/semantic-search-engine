# WordPress Integration for SEED

To integrate SEED Semantic Search with WordPress:

## Integration via PHP

Add the following to your `functions.php` or a custom plugin:

```php
function call_seed_search($query) {
    $url = 'http://seed-gateway:8000/v1/search/semantic';
    $api_key = 'YOUR_SECRET_TOKEN';

    $response = wp_remote_post($url, [
        'headers' => [
            'Authorization' => 'Bearer ' . $api_key,
            'Content-Type' => 'application/json',
        ],
        'body' => json_encode([
            'query' => $query,
        ]),
    ]);

    if (is_wp_error($response)) {
        return 'Search failed.';
    }

    $body = json_decode(wp_remote_retrieve_body($response), true);
    return $body['answer'];
}
```

## Security Note
Always call the SEED gateway **server-to-server**. Do not expose the Bearer token or the SEED URL in client-side JavaScript.
",filePath:"/Users/mmh/Documents/data_science_kdf/integrations/wordpress/README.md"}end"filePath: