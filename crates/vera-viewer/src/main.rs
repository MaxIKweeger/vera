//! vera-viewer — interactive astronomical source viewer (eframe 0.35 + egui 0.35)
//!
//! Usage: vera-viewer <brick.fits.fz> [band]

use std::f32::consts::PI;
use std::path::PathBuf;

use eframe::egui;
use ndarray::Array2;
use vera_fits::{read_image_f32, WcsHeader};
use vera_pipeline::background::{BackgroundConfig, BackgroundMap};
use vera_pipeline::detect::{detect, DetectConfig};
use vera_pipeline::measure::{measure_all, Measurement, MeasureConfig};

// ── ZScale / stretch ──────────────────────────────────────────────────────────

fn zscale(image: &Array2<f32>) -> (f32, f32) {
    const MAX_SAMPLES: usize = 12_000;
    let total = image.len();
    let step = (total / MAX_SAMPLES).max(1);
    let mut s: Vec<f32> = image.iter()
        .step_by(step)
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if s.is_empty() { return (0.0, 1.0); }
    s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    (s[(n / 200).max(0)], s[(n * 199 / 200).min(n - 1)])
}

fn apply_stretch(v: f32, vmin: f32, vmax: f32, a: f32) -> u8 {
    let x = ((v - vmin) / (vmax - vmin).max(1e-10)).clamp(-0.5, 10.0) * a;
    let norm  = (x + (x * x + 1.0).sqrt()).ln();
    let scale = (a + (a * a + 1.0).sqrt()).ln().max(1e-10);
    ((norm / scale).clamp(0.0, 1.0) * 255.0) as u8
}

// ── App state ─────────────────────────────────────────────────────────────────

struct VeraApp {
    image:        Array2<f32>,
    wcs:          Option<WcsHeader>,
    measurements: Vec<Measurement>,
    img_w:        usize,
    img_h:        usize,
    brick:        String,
    band:         String,

    texture:       Option<egui::TextureHandle>,
    texture_dirty: bool,

    zoom:        f32,
    pan:         egui::Vec2,
    initialized: bool,

    vmin:         f32,
    vmax:         f32,
    stretch_a:    f32,
    show_sources: bool,
    selected:     Option<usize>,
}

impl VeraApp {
    fn new(
        _cc: &eframe::CreationContext<'_>,
        image: Array2<f32>,
        wcs: Option<WcsHeader>,
        measurements: Vec<Measurement>,
        vmin: f32, vmax: f32,
        brick: String, band: String,
    ) -> Self {
        let (h, w) = image.dim();
        Self {
            img_w: w, img_h: h,
            pan: egui::vec2(w as f32 / 2.0, h as f32 / 2.0),
            zoom: 0.0, initialized: false,
            texture: None, texture_dirty: true,
            vmin, vmax, stretch_a: 5.0,
            show_sources: true, selected: None,
            image, wcs, measurements, brick, band,
        }
    }

    fn rebuild_texture(&mut self, ctx: &egui::Context) {
        let (h, w) = self.image.dim();
        let vmin = self.vmin;
        let vmax = self.vmax;
        let a    = self.stretch_a;
        let mut rgba = Vec::with_capacity(w * h * 4);
        for &v in self.image.iter() {
            let g = apply_stretch(v, vmin, vmax, a);
            rgba.push(g); rgba.push(g); rgba.push(g); rgba.push(255);
        }
        let ci = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
        let opts = egui::TextureOptions {
            magnification: egui::TextureFilter::Nearest,
            minification:  egui::TextureFilter::Linear,
            ..Default::default()
        };
        if let Some(t) = &mut self.texture {
            t.set(ci, opts);
        } else {
            self.texture = Some(ctx.load_texture("vera-image", ci, opts));
        }
        self.texture_dirty = false;
    }

    fn img_to_screen(&self, ix: f32, iy: f32, vp: egui::Rect) -> egui::Pos2 {
        let c = vp.center();
        egui::pos2(c.x + (ix - self.pan.x) * self.zoom,
                   c.y + (iy - self.pan.y) * self.zoom)
    }

    fn screen_to_img(&self, sp: egui::Pos2, vp: egui::Rect) -> (f32, f32) {
        let c = vp.center();
        (self.pan.x + (sp.x - c.x) / self.zoom,
         self.pan.y + (sp.y - c.y) / self.zoom)
    }

    fn show_controls(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Vera");
            ui.label(format!("Brick  : {}", self.brick));
            ui.label(format!("Band   : {}", self.band));
            ui.label(format!("Image  : {}×{} px", self.img_w, self.img_h));
            ui.label(format!("Sources: {}", self.measurements.len()));
            ui.separator();

            ui.label("Stretch (asinh)");
            let da = ui.add(egui::Slider::new(&mut self.stretch_a, 0.5..=50.0).text("a")).changed();
            let dl = ui.add(egui::Slider::new(&mut self.vmin, -1.0..=(self.vmax - 0.01)).text("vmin")).changed();
            let dh = ui.add(egui::Slider::new(&mut self.vmax, (self.vmin + 0.01)..=2000.0).text("vmax")).changed();
            if da || dl || dh { self.texture_dirty = true; }

            if ui.button("ZScale reset").clicked() {
                (self.vmin, self.vmax) = zscale(&self.image);
                self.texture_dirty = true;
            }
            ui.separator();

            ui.checkbox(&mut self.show_sources, "Show sources");
            ui.separator();

            ui.label("Navigation");
            ui.small("Drag       : pan");
            ui.small("Scroll     : zoom");
            ui.small("Dbl-click  : fit to window");
            ui.separator();

            if let Some(idx) = self.selected {
                if let Some(m) = self.measurements.get(idx) {
                    ui.label("── Selected source ──");
                    ui.label(format!("RA  : {:.5}°",  m.ra.unwrap_or(f64::NAN)));
                    ui.label(format!("Dec : {:.5}°",  m.dec.unwrap_or(f64::NAN)));
                    ui.label(format!("X   : {:.1} px", m.x_c));
                    ui.label(format!("Y   : {:.1} px", m.y_c));
                    ui.label(format!("flux_auto : {:.3} nmg", m.flux_auto));
                    ui.label(format!("a={:.1}  b={:.1}  θ={:.1}°", m.a, m.b, m.theta));
                    ui.label(format!("Kron r : {:.2}", m.kron_radius));
                    ui.label(format!("Npix : {}", m.npix));
                    ui.label(format!("Flags : 0x{:02X}", m.flags));
                }
            }
        });
    }

    fn show_image_panel(&mut self, ui: &mut egui::Ui) {
        let vp = ui.available_rect_before_wrap();

        if !self.initialized {
            let zoom_x = vp.width()  / self.img_w as f32;
            let zoom_y = vp.height() / self.img_h as f32;
            self.zoom = zoom_x.min(zoom_y);
            self.initialized = true;
        }

        let response = ui.allocate_rect(vp, egui::Sense::click_and_drag());

        if response.dragged() {
            let d = response.drag_delta();
            self.pan -= egui::vec2(d.x / self.zoom, d.y / self.zoom);
        }

        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            let factor = 1.08f32.powf(scroll / 20.0);
            if let Some(hp) = response.hover_pos() {
                let (ix0, iy0) = self.screen_to_img(hp, vp);
                self.zoom = (self.zoom * factor).clamp(0.02, 100.0);
                let (ix1, iy1) = self.screen_to_img(hp, vp);
                self.pan.x -= ix1 - ix0;
                self.pan.y -= iy1 - iy0;
            } else {
                self.zoom = (self.zoom * factor).clamp(0.02, 100.0);
            }
        }

        if response.double_clicked() {
            let zoom_x = vp.width()  / self.img_w as f32;
            let zoom_y = vp.height() / self.img_h as f32;
            self.zoom = zoom_x.min(zoom_y);
            self.pan  = egui::vec2(self.img_w as f32 / 2.0, self.img_h as f32 / 2.0);
        }

        if let Some(texture) = &self.texture {
            let iw = self.img_w as f32;
            let ih = self.img_h as f32;
            let c  = vp.center();
            let u_min = (self.pan.x + (vp.left()   - c.x) / self.zoom) / iw;
            let u_max = (self.pan.x + (vp.right()  - c.x) / self.zoom) / iw;
            let v_min = (self.pan.y + (vp.top()    - c.y) / self.zoom) / ih;
            let v_max = (self.pan.y + (vp.bottom() - c.y) / self.zoom) / ih;
            let uv = egui::Rect::from_min_max(egui::pos2(u_min, v_min), egui::pos2(u_max, v_max));

            let painter = ui.painter_at(vp);
            painter.image(texture.id(), vp, uv, egui::Color32::WHITE);

            if self.show_sources {
                let expanded_vp = vp.expand(60.0);
                for (idx, m) in self.measurements.iter().enumerate() {
                    let sc = self.img_to_screen(m.x_c as f32, m.y_c as f32, vp);
                    if !expanded_vp.contains(sc) { continue; }

                    let color = if Some(idx) == self.selected {
                        egui::Color32::YELLOW
                    } else if m.flags != 0 {
                        egui::Color32::from_rgb(255, 120, 0)
                    } else {
                        egui::Color32::from_rgb(0, 220, 80)
                    };

                    let a = m.a.max(2.0);
                    let b = m.b.max(1.0);
                    let cos_t = (m.theta * PI / 180.0).cos();
                    let sin_t = (m.theta * PI / 180.0).sin();
                    let mut pts: Vec<egui::Pos2> = (0..32)
                        .map(|i| {
                            let t  = 2.0 * PI * i as f32 / 32.0;
                            let ex = a * t.cos();
                            let ey = b * t.sin();
                            let rx = ex * cos_t - ey * sin_t;
                            let ry = ex * sin_t + ey * cos_t;
                            self.img_to_screen(m.x_c as f32 + rx, m.y_c as f32 + ry, vp)
                        })
                        .collect();
                    pts.push(pts[0]);
                    let sw = if self.zoom > 3.0 { 1.5 } else { 1.0 };
                    painter.add(egui::Shape::line(pts, egui::Stroke::new(sw, color)));
                }
            }

            // Click → select nearest source.
            if response.clicked() {
                if let Some(cp) = response.interact_pointer_pos() {
                    let (cx, cy) = self.screen_to_img(cp, vp);
                    let nearest = self.measurements.iter().enumerate().min_by(|(_, a), (_, b)| {
                        let da = (a.x_c as f32 - cx).hypot(a.y_c as f32 - cy);
                        let db = (b.x_c as f32 - cx).hypot(b.y_c as f32 - cy);
                        da.partial_cmp(&db).unwrap()
                    });
                    if let Some((idx, m)) = nearest {
                        if (m.x_c as f32 - cx).hypot(m.y_c as f32 - cy) * self.zoom < 50.0 {
                            self.selected = Some(idx);
                        } else {
                            self.selected = None;
                        }
                    }
                }
            }

            // Coordinate display.
            if let Some(hp) = response.hover_pos() {
                let (ix, iy) = self.screen_to_img(hp, vp);
                let txt = if let Some(wcs) = &self.wcs {
                    let (ra, dec) = wcs.pix_to_sky(ix as f64, iy as f64);
                    format!("({:.0}, {:.0}) px  RA {ra:.5}°  Dec {dec:+.5}°", ix, iy)
                } else {
                    format!("({:.0}, {:.0}) px", ix, iy)
                };
                painter.text(
                    vp.left_bottom() + egui::vec2(6.0, -6.0),
                    egui::Align2::LEFT_BOTTOM,
                    txt,
                    egui::FontId::monospace(11.0),
                    egui::Color32::from_rgba_premultiplied(255, 255, 180, 220),
                );
            }
        }
    }
}

impl eframe::App for VeraApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.texture_dirty {
            let ctx = ui.ctx().clone();
            self.rebuild_texture(&ctx);
        }

        let full_rect = ui.max_rect();
        let left_w    = 230.0;
        let sep_w     = 1.0;

        // Controls panel
        let left_rect = egui::Rect::from_min_size(full_rect.min, egui::vec2(left_w, full_rect.height()));
        let mut left_ui = ui.new_child(egui::UiBuilder::new().max_rect(left_rect));
        egui::Frame::default()
            .fill(ui.visuals().panel_fill)
            .show(&mut left_ui, |ui| self.show_controls(ui));

        // Separator
        let sep_rect = egui::Rect::from_min_size(
            egui::pos2(full_rect.min.x + left_w, full_rect.min.y),
            egui::vec2(sep_w, full_rect.height()),
        );
        ui.painter().rect_filled(sep_rect, 0.0, ui.visuals().window_stroke.color);

        // Image panel
        let right_rect = egui::Rect::from_min_max(
            egui::pos2(full_rect.min.x + left_w + sep_w, full_rect.min.y),
            full_rect.max,
        );
        let mut right_ui = ui.new_child(egui::UiBuilder::new().max_rect(right_rect));
        self.show_image_panel(&mut right_ui);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path_str = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("Usage: vera-viewer <brick.fits.fz> [band]");
        std::process::exit(1);
    });
    let band  = args.get(2).cloned().unwrap_or_else(|| "r".into());
    let path  = PathBuf::from(&path_str);
    let stem  = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
    let brick = stem.split('-').nth(1).unwrap_or(stem).to_string();

    println!("Vera viewer — loading {path_str} …");
    let t0 = std::time::Instant::now();

    let (image, wcs) = read_image_f32(&path).unwrap_or_else(|e| {
        eprintln!("FITS error: {e}");
        std::process::exit(1);
    });
    let bg   = BackgroundMap::estimate(&image, &BackgroundConfig::default());
    let det  = detect(&image, &bg, &DetectConfig::default());
    let meas = measure_all(&image, &bg, &det, wcs.as_ref(), &MeasureConfig::default());
    let (vmin, vmax) = zscale(&image);

    println!("Pipeline : {:.1?}  ({} sources)", t0.elapsed(), meas.len());
    println!("ZScale   : vmin={vmin:.4}  vmax={vmax:.4}");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(format!("Vera — {brick} ({band})"))
            .with_inner_size([1400.0, 900.0]),
        ..Default::default()
    };

    eframe::run_native(
        "vera-viewer",
        options,
        Box::new(move |cc| Ok(Box::new(VeraApp::new(cc, image, wcs, meas, vmin, vmax, brick, band)))),
    ).unwrap();
}
