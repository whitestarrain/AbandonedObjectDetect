drop database abandoned_object_detect;
create database abandoned_object_detect;
use abandoned_object_detect;
CREATE TABLE user
(
    id       INT PRIMARY KEY AUTO_INCREMENT, -- 用户id
    username VARCHAR(20),                    -- 登录名称
    password VARCHAR(20),                     -- 密码
    insert_time datetime not null,
    update_time datetime not null
); -- 用户表


INSERT INTO user(username,PASSWORD,insert_time,update_time) VALUES('root','root',NOW(),NOW());

create table video_resource
(
    id          INT PRIMARY KEY AUTO_INCREMENT,
    file_name   varchar(20)  not null,
    source_path varchar(100) not null,
    type        int          not null default 0, -- 视频源类型。 0 文件 1 摄像头
    insert_time datetime     not null,
    update_time datetime     not null
);

insert into video_resource(file_name, source_path, type, insert_time, update_time)
values ('test_video', 'D:/MyRepo/AbandonedObjectDetect/datasets/test_dataset/1.avi', 0, now(), now());

insert into video_resource(file_name, source_path, type, insert_time, update_time)
values ('camera0', '0', 1, now(), now());
