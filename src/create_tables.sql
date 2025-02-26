CREATE TABLE IF NOT EXISTS pets (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    pet_type VARCHAR(255) NOT NULL,
    affected_limb VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS converters (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    path_image VARCHAR(255),
    path_video VARCHAR(255) NOT NULL,
    path_obj VARCHAR(255),
    status VARCHAR(255) NOT NULL DEFAULT 'processing'
); 